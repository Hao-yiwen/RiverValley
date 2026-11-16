"""只在本机提供 RiverValley 聊天页面，加载冻结源码和已交付的 SFT 权重。"""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import queue
import re
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger('rivervalley.chat')


def validate_request(body):
    """约束输入结构和生成上限，拒绝无效请求而不修改模型默认配置。"""
    if not isinstance(body, dict):
        raise ValueError('请求必须是 JSON 对象')
    request_id = body.get('request_id', '')
    if not isinstance(request_id, str) or not re.fullmatch(r'[a-zA-Z0-9-]{8,64}', request_id):
        raise ValueError('请求标识无效，请刷新页面重试')
    messages = body.get('messages')
    if not isinstance(messages, list) or not 1 <= len(messages) <= 101:
        raise ValueError('请发送 1 至 101 条对话消息，过长时请新建对话')
    cleaned = []
    for index, item in enumerate(messages):
        role = 'user' if index % 2 == 0 else 'assistant'
        if not isinstance(item, dict) or item.get('role') != role:
            raise ValueError('对话必须按用户、助手交替排列')
        content = item.get('content')
        if not isinstance(content, str) or not content.strip() or len(content) > 16000:
            raise ValueError('消息不能为空，且单条不能超过 16000 个字符')
        cleaned.append({'role': role, 'content': content})
    if cleaned[-1]['role'] != 'user':
        raise ValueError('最后一条必须是你的问题')
    length = body.get('max_tokens', 128)
    if type(length) is not int or not 16 <= length <= 256:
        raise ValueError('回答长度必须在 16 至 256 token 之间')
    temperature = body.get('temperature', 0)
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 1.5:
        raise ValueError('温度必须在 0 至 1.5 之间')
    repetition = body.get('repetition_penalty', 1.0)
    if type(repetition) not in (int, float) or not math.isfinite(repetition) or not 1 <= repetition <= 1.5:
        raise ValueError('重复惩罚必须在 1 至 1.5 之间')
    return request_id, cleaned, length, float(temperature), float(repetition)


def fit_context(tokenizer, messages, max_tokens, context_limit):
    """只移除完整的最早对话轮次，为回答预留空间；不截断当前问题。"""
    kept = list(messages)
    dropped = 0
    while True:
        ids = tokenizer.apply_chat_template(kept, add_generation_prompt=True, tokenize=True)
        if len(ids) + max_tokens <= context_limit:
            return ids, dropped
        if len(kept) <= 1:
            raise ValueError(f'当前问题太长，请缩短问题或减少回答长度（上下文上限 {context_limit} token）')
        kept = kept[2:]
        dropped += 1


class ChatEngine:
    def __init__(self, model_dir, code_root, device):
        self.model_dir = Path(model_dir).resolve()
        self.code_root = Path(code_root).resolve()
        self.requested_device = device
        self.device = None
        self.state = 'loading'
        self.error = None
        self.model = None
        self.lock = threading.Lock()
        self.active = None
        self.context_limit = 1024
        self.receipt = {}
        receipt_path = self.model_dir.parent / 'delivery_receipt.json'
        if receipt_path.exists():
            self.receipt = json.loads(receipt_path.read_text(encoding='utf-8'))

    def load(self):
        started = time.monotonic()
        try:
            # 使用与训练一起冻结的模块，不从网络下载模型或执行远端源码。
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            sys.path.insert(0, str(self.code_root / 'src'))
            import torch
            from transformers import PreTrainedTokenizerFast
            from model import RiverValleyModel
            self.torch = torch
            device = self.requested_device
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
            logger.info('加载本地模型：目录=%s，源码=%s，设备=%s', self.model_dir, self.code_root, device)
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.model_dir, local_files_only=True)
            self.model = RiverValleyModel.from_pretrained(self.model_dir, dtype=torch.float32,
                                                         local_files_only=True).eval().to(device)
            self.device = device
            self.context_limit = self.model.config.max_seq_len
            self.eos_ids = list({self.tokenizer.eos_token_id,
                                 self.tokenizer.convert_tokens_to_ids('<|im_end|>')} - {None})
            self.state = 'ready'
            logger.info('模型就绪：设备=%s，参数=%d，耗时=%.2f秒', device,
                        sum(p.numel() for p in self.model.parameters()), time.monotonic() - started)
        except Exception as error:
            self.state = 'error'
            self.error = f'{type(error).__name__}: {error}'
            logger.exception('本地模型加载失败')

    def status(self):
        with self.lock:
            return {'state': self.state, 'error': self.error, 'device': self.device,
                    'busy': self.active is not None, 'model_name': 'RiverValley 1B · SFT',
                    'context_limit': self.context_limit, 'precision': 'FP32',
                    'sft_step': self.receipt.get('best_sft_step')}

    def start(self, body):
        request_id, messages, max_tokens, temperature, repetition = validate_request(body)
        with self.lock:
            if self.state != 'ready':
                raise RuntimeError('模型尚未就绪，请等待加载完成')
            if self.active is not None:
                raise RuntimeError('另一个回答正在生成，请稍后重试')
            ids, dropped = fit_context(self.tokenizer, messages, max_tokens, self.context_limit)
            job = {'id': request_id, 'events': queue.Queue(), 'stop': threading.Event()}
            self.active = job
            worker = threading.Thread(target=self._generate, args=(job, ids, dropped, max_tokens, temperature, repetition), daemon=True)
            worker.start()
            return job

    def cancel(self, request_id):
        with self.lock:
            if self.active and self.active['id'] == request_id:
                self.active['stop'].set()
                logger.info('请求停止生成：id=%s', request_id)
                return True
            return False

    def _generate(self, job, ids, dropped, max_tokens, temperature, repetition):
        from transformers import TextStreamer, StoppingCriteria
        started = time.monotonic()
        first_text_at = None
        events = job['events']
        result = None
        event_name = 'done'

        class Streamer(TextStreamer):
            def on_finalized_text(self, text, stream_end=False):
                nonlocal first_text_at
                if text:
                    if first_text_at is None:
                        first_text_at = time.monotonic()
                    events.put(('delta', {'text': text}))

        class StopRequested(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return job['stop'].is_set()

        try:
            logger.info('生成开始：id=%s，输入token=%d，最大输出=%d，温度=%.2f，重复惩罚=%.2f，移除历史轮次=%d',
                        job['id'], len(ids), max_tokens, temperature, repetition, dropped)
            events.put(('meta', {'input_tokens': len(ids), 'dropped_turns': dropped, 'device': self.device,
                                'max_tokens': max_tokens, 'temperature': temperature, 'repetition_penalty': repetition}))
            inputs = self.torch.tensor([ids], dtype=self.torch.long, device=self.device)
            kwargs = {'input_ids': inputs, 'attention_mask': self.torch.ones_like(inputs),
                      'max_new_tokens': max_tokens, 'do_sample': temperature > 0,
                      'repetition_penalty': repetition, 'eos_token_id': self.eos_ids,
                      'top_p': 0.9 if temperature > 0 else 1.0,
                      'pad_token_id': self.tokenizer.pad_token_id, 'use_cache': True,
                      'streamer': Streamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True),
                      'stopping_criteria': [StopRequested()]}
            if temperature > 0:
                kwargs.update(temperature=temperature, top_p=0.9)
            with self.torch.inference_mode():
                output = self.model.generate(**kwargs)
            count = output.shape[1] - len(ids)
            elapsed = time.monotonic() - started
            reason = 'cancelled' if job['stop'].is_set() else ('length' if count == max_tokens else 'stop')
            result = {'output_tokens': count, 'elapsed_seconds': round(elapsed, 2),
                      'tokens_per_second': round(count / max(elapsed, 0.001), 2),
                      'first_text_seconds': round(first_text_at - started, 2) if first_text_at else None,
                      'finish_reason': reason}
            logger.info('生成结束：id=%s，输出token=%d，耗时=%.2f秒，原因=%s', job['id'], count, elapsed, reason)
        except Exception as error:
            logger.exception('生成失败：id=%s，设备=%s', job['id'], self.device)
            event_name = 'error'
            result = {'message': f'本地推理失败：{type(error).__name__}: {error}'}
        finally:
            with self.lock:
                self.active = None
            # 先释放模型占用标志，再通知前端本次请求结束。
            events.put((event_name, result))


class ChatHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        logger.info('本地HTTP：%s', fmt % args)

    def allowed(self):
        port = self.server.server_port
        hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
        origin = self.headers.get('Origin')
        return self.headers.get('Host') in hosts and (not origin or origin in {f'http://{h}' for h in hosts})

    def send_json(self, code, data):
        raw = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if not self.allowed():
            return self.send_json(403, {'error': '只接受本机页面请求'})
        if self.path == '/api/status':
            return self.send_json(200, self.server.engine.status())
        if self.path != '/':
            return self.send_json(404, {'error': '页面不存在'})
        raw = (ROOT / 'src/web/chat.html').read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        if not self.allowed():
            return self.send_json(403, {'error': '只接受本机页面请求'})
        self.close_connection = True
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 131072:
                raise ValueError('请求为空或过大')
            if self.headers.get_content_type() != 'application/json':
                raise ValueError('请使用 JSON 请求')
            body = json.loads(self.rfile.read(length))
            if self.path == '/api/version':
                if not isinstance(body, dict) or not isinstance(body.get('version'), str):
                    raise ValueError('版本选择格式错误')
                if not hasattr(self.server.engine, 'switch'):
                    raise ValueError('自定义权重模式不支持页面切换版本')
                self.server.engine.switch(body['version'])
                return self.send_json(200, self.server.engine.status())
            if self.path == '/api/cancel':
                if not isinstance(body, dict):
                    raise ValueError('停止请求格式错误')
                return self.send_json(200, {'cancelled': self.server.engine.cancel(body.get('request_id'))})
            if self.path != '/api/chat':
                return self.send_json(404, {'error': '接口不存在'})
            job = self.server.engine.start(body)
        except (ValueError, UnicodeDecodeError) as error:
            logger.warning('请求验证失败：路径=%s，原因=%s', self.path, error)
            return self.send_json(400, {'error': str(error)})
        except RuntimeError as error:
            return self.send_json(409, {'error': str(error)})
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'close')
            self.end_headers()
            while True:
                try:
                    event, payload = job['events'].get(timeout=1)
                    packet = f'event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n'
                except queue.Empty:
                    packet = ': 等待生成\n\n'
                    event = 'ping'
                self.wfile.write(packet.encode('utf-8'))
                self.wfile.flush()
                if event in ('done', 'error'):
                    break
        except (BrokenPipeError, ConnectionResetError, OSError):
            logger.info('页面断开连接，停止本次生成：id=%s', job['id'])
            self.server.engine.cancel(job['id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--model', help='自定义模式的权重目录，必须同时指定冻结源码')
    parser.add_argument('--code-root', help='自定义模式的冻结源码目录')
    parser.add_argument('--version', help='已交付版本目录名；默认选择最近完成的版本')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    args = parser.parse_args()
    os.chdir(ROOT)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if bool(args.model) != bool(args.code_root):
        parser.error('--model 与 --code-root 必须同时指定')
    if args.model:
        engine = ChatEngine(args.model, args.code_root, args.device)
        threading.Thread(target=engine.load, daemon=True).start()
    else:
        from chat_versions import VersionedEngine
        engine = VersionedEngine(ROOT / 'checkpoints', args.device, args.version)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), ChatHandler)
    server.daemon_threads = True
    server.engine = engine
    # 后台服务收到常规终止信号时，也要释放独立推理进程及其模型内存。
    def stop_server(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_server)
    logger.info('聊天服务启动：http://127.0.0.1:%d，仅监听本机', args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info('收到退出信号，关闭本地聊天服务')
    finally:
        if engine.active:
            engine.cancel(engine.active['id'])
        if hasattr(engine, 'close'):
            engine.close()
        server.server_close()


if __name__ == '__main__':
    main()
