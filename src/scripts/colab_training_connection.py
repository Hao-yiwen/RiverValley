"""本机持续接收真实训练单元格输出，断线后重新连接同一个内核。"""
import argparse
import json
import logging
from pathlib import Path
import time
from urllib.parse import urlencode
import uuid

logger = logging.getLogger(__name__)


def consume(session_name, kernel_id, endpoint, output_dir, stop_at,
            get_session, connect, clock=time.time, sleep=time.sleep):
    """只订阅已有内核的输出；不发送执行请求，不创建或重启实例。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client_id = str(uuid.uuid4())
    attempts = 0
    while clock() < stop_at:
        socket = None
        try:
            session = get_session(session_name, ignore_missing_session=True)
            if session is None or session.endpoint != endpoint:
                logger.info('原训练实例已消失或被替换，结束输出订阅：%s', session_name)
                return
            # 每次重连通过 CLI 刷新到期凭据，不复用旧 WebSocket 地址。
            url = session.url.rstrip('/').replace('https://', 'wss://').replace('http://', 'ws://')
            url += '/api/kernels/' + kernel_id + '/channels?'
            url += urlencode({'session_id': client_id, 'colab-runtime-proxy-token': session.token})
            socket = connect(url, header={'X-Colab-Client-Agent': 'colab-cli',
                             'X-Colab-Runtime-Proxy-Token': session.token}, timeout=45)
            attempts += 1
            logger.info('训练输出连接已建立：session=%s，kernel=%s，连接次数=%d',
                        session_name, kernel_id, attempts)
            while clock() < stop_at:
                try:
                    raw = socket.recv()
                except TimeoutError:
                    continue
                if not raw:
                    raise ConnectionError('服务器关闭输出连接')
                if isinstance(raw, bytes):
                    # 普通训练日志为 JSON 文本，二进制显示内容不写入日志。
                    continue
                message = json.loads(raw)
                kind = message.get('msg_type') or message.get('header', {}).get('msg_type')
                content = message.get('content', {})
                if kind == 'stream':
                    with (output_dir / 'training_stream.log').open('a', encoding='utf-8') as log:
                        log.write(content.get('text', ''))
                heartbeat = {'time': clock(), 'session': session_name, 'kernel_id': kernel_id,
                             'connections': attempts, 'last_message_type': kind,
                             'execution_state': content.get('execution_state')}
                (output_dir / 'connection_heartbeat.json').write_text(
                    json.dumps(heartbeat, ensure_ascii=False), encoding='utf-8')
        except Exception as error:
            # 异常字符串可能带凭据 URL，只记录类型；不把瞬时断线当作训练退出。
            logger.warning('训练输出连接中断，稍后重连原内核：session=%s，类型=%s',
                           session_name, type(error).__name__)
        finally:
            if socket is not None:
                socket.close()
        sleep(5)
    logger.info('达到本轮硬截止，结束训练输出订阅')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--stop-at', required=True, type=float)
    args = parser.parse_args()
    from colab_cli.common import state
    import websocket
    session = state.get_session(args.session)
    if not session.kernel_id:
        raise RuntimeError('缺少已有训练内核 ID，拒绝创建新内核')
    # 包装超时，让空闲的输出间隔仍等待同一实际训练任务。
    def connect(*a, **kw):
        ws = websocket.create_connection(*a, **kw)
        original_recv = ws.recv
        def recv():
            try:
                return original_recv()
            except websocket.WebSocketTimeoutException as error:
                raise TimeoutError from error
        ws.recv = recv
        return ws
    consume(args.session, session.kernel_id, session.endpoint, args.output_dir,
            args.stop_at, state.get_session, connect)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # 第三方网络日志可能打印带凭据的地址，诊断由本模块提供。
    logging.getLogger('websocket').setLevel(logging.CRITICAL)
    main()
