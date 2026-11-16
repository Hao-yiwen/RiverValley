"""每次仅加载一个交付版本，用独立进程隔离各版本的冻结模型源码。"""
import hashlib
import json
import logging
import multiprocessing
from pathlib import Path
import queue
import threading
import time

log = logging.getLogger('rivervalley.chat.versions')


def discover_deliveries(root):
    versions = {}
    for receipt_path in sorted(Path(root).glob('delivery*/delivery_receipt.json')):
        try:
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            folder = receipt_path.parent.resolve()
            model, code = folder / 'sft_best', folder / 'code/river-valley'
            if receipt.get('status') != 'weights_downloaded_verified' or len(receipt.get('sha256', '')) != 64:
                continue
            if not all(p.exists() for p in [model / 'model.safetensors', model / 'tokenizer.json',
                                           model / 'config.json', code / 'src/model/gpt.py']):
                continue
            versions[folder.name] = {'id': folder.name, 'label': receipt.get('label', folder.name.replace('delivery_', '')),
                'time': receipt.get('time', 0), 'model_dir': str(model), 'code_root': str(code),
                'sha256': receipt['sha256'], 'sft_step': receipt.get('best_sft_step'),
                'pretrain_tokens': receipt.get('pretrain_tokens'), 'quality_status': receipt.get('quality_status', '请实测回答效果')}
        except (OSError, ValueError, TypeError):
            log.warning('跳过不完整交付回执：%s', receipt_path)
    return versions


def model_worker(spec, device, commands, results):
    """冻结源码只在此进程中导入，切换版本时退出整个进程后再加载下一版。"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    from local_chat import ChatEngine
    engine = ChatEngine(spec['model_dir'], spec['code_root'], device)
    try:
        path = Path(spec['model_dir']) / 'model.safetensors'
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != spec['sha256']:
                raise ValueError('权重 SHA256 与交付回执不符')
        engine.load()
    except Exception as error:
        log.exception('版本加载前校验失败：%s', spec['id'])
        engine.state, engine.error = 'error', str(error)
    results.put(('status', engine.status()))
    job = None
    while True:
        try:
            command, body = commands.get(timeout=.1)
        except queue.Empty:
            command, body = None, None
        if command == 'shutdown':
            if job:
                engine.cancel(job['id'])
            return
        if command == 'start':
            try:
                job = engine.start(body)
            except Exception as error:
                results.put(('event', (body['request_id'], 'error', {'message': str(error)})))
        elif command == 'cancel':
            engine.cancel(body)
        if job:
            while True:
                try:
                    event, data = job['events'].get_nowait()
                except queue.Empty:
                    break
                results.put(('event', (job['id'], event, data)))
                if event in ('done', 'error'):
                    job = None
                    break


class VersionedEngine:
    def __init__(self, root, device, selected=None):
        self.root, self.device = Path(root), device
        self.versions = discover_deliveries(root)
        if not self.versions:
            raise ValueError('没有已校验的本地交付版本')
        selected = selected or max(self.versions, key=lambda key: self.versions[key]['time'])
        if selected not in self.versions:
            raise ValueError('交付版本不存在：' + selected)
        self.lock = threading.RLock()
        self.context = multiprocessing.get_context('spawn')
        self.process = self.commands = None
        self.active = None
        self.closed = False
        self.current = selected
        self.cached = {'state': 'loading', 'error': None, 'device': None}
        threading.Thread(target=self._replace, args=(self.versions[selected],), daemon=True).start()

    def _replace(self, spec):
        try:
            old = self.process
            if old:
                self.commands.put(('shutdown', None))
                old.join(timeout=5)
                if old.is_alive():
                    old.terminate()
                    old.join(timeout=5)
                if old.is_alive():
                    raise RuntimeError('旧推理进程尚未释放，禁止加载第二个模型')
            with self.lock:
                if self.closed:
                    return
                commands, results = self.context.Queue(), self.context.Queue()
                process = self.context.Process(target=model_worker, args=(spec, self.device, commands, results), daemon=True)
                self.cached = {'state': 'loading', 'error': None, 'device': None}
                self.commands, self.process = commands, process
                process.start()
            log.info('加载交付版本：版本=%s 推理进程=%s', spec['id'], process.pid)
            while not self.closed:
                try:
                    kind, value = results.get(timeout=.5)
                except queue.Empty:
                    with self.lock:
                        if process is not self.process or self.cached['state'] == 'switching':
                            return
                        if not process.is_alive():
                            raise RuntimeError(f'本地推理进程退出，退出码={process.exitcode}')
                    continue
                with self.lock:
                    if process is not self.process or self.cached['state'] == 'switching':
                        return
                    if kind == 'status':
                        self.cached = value
                    elif kind == 'event':
                        request_id, event, data = value
                        if self.active and request_id == self.active['id']:
                            job = self.active
                            if event in ('done', 'error'):
                                self.active = None
                            job['events'].put((event, data))
        except Exception as error:
            log.exception('版本推理进程失败：%s', spec['id'])
            with self.lock:
                if spec['id'] != self.current:
                    return
                self.cached.update(state='error', error=str(error))
                if self.active:
                    self.active['events'].put(('error', {'message': str(error)}))
                    self.active = None

    def status(self):
        with self.lock:
            self.versions.update(discover_deliveries(self.root))
            spec = self.versions[self.current]
            return dict(self.cached, busy=self.active is not None, version=self.current,
                        model_name='RiverValley 1B · ' + spec['label'], sft_step=spec['sft_step'],
                        quality_status=spec['quality_status'],
                        versions=[{k: v[k] for k in ('id', 'label', 'sft_step')} for v in self.versions.values()])

    def switch(self, version):
        with self.lock:
            self.versions.update(discover_deliveries(self.root))
            if version not in self.versions:
                raise ValueError('所选版本尚未完整下载校验')
            if self.active or self.cached['state'] in ('loading', 'switching'):
                raise RuntimeError('请等待当前加载或回答结束再切换版本')
            if version == self.current and self.cached['state'] == 'ready':
                return
            self.cached.update(state='switching', error=None)
            self.current = version
            threading.Thread(target=self._replace, args=(self.versions[version],), daemon=True).start()

    def start(self, body):
        from local_chat import validate_request
        request_id, *_ = validate_request(body)
        with self.lock:
            if self.cached['state'] != 'ready':
                raise RuntimeError('模型尚未就绪，请等待加载完成')
            if self.active:
                raise RuntimeError('另一个回答正在生成，请稍后重试')
            self.active = {'id': request_id, 'events': queue.Queue()}
            self.commands.put(('start', body))
            return self.active

    def cancel(self, request_id):
        with self.lock:
            if self.active and self.active['id'] == request_id:
                self.commands.put(('cancel', request_id))
                return True
            return False

    def close(self):
        with self.lock:
            self.closed = True
            if self.commands:
                self.commands.put(('shutdown', None))
            process = self.process
        if process:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
