"""直接上传 Drive 并校验服务器摘要；不把挂载盘缓存当成持久化完成。"""
import hashlib
import json
import logging
import os
from pathlib import Path
import time

import requests

logger = logging.getLogger(__name__)
FIELDS = 'id,name,size,md5Checksum,sha256Checksum'


class DriveRequestError(RuntimeError):
    def __init__(self, status, reasons):
        self.status, self.reasons = status, reasons
        super().__init__(f'Drive 请求失败：HTTP {status}，原因={",".join(reasons) or "unknown"}')


def fingerprints(path):
    digests = {name: hashlib.new(name) for name in ('sha256', 'md5')}
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            for digest in digests.values():
                digest.update(block)
    return {'bytes': Path(path).stat().st_size,
            **{name: digest.hexdigest() for name, digest in digests.items()}}


def check_remote(metadata, expected):
    """服务器必须返回完整大小及内容摘要；缺失摘要也不能放行。"""
    if int(metadata.get('size', -1)) != expected['bytes']:
        raise ValueError('Drive 服务器文件大小不匹配')
    if metadata.get('sha256Checksum'):
        valid = metadata['sha256Checksum'] == expected['sha256']
    else:
        valid = bool(expected.get('md5')) and metadata.get('md5Checksum') == expected['md5']
    if not valid:
        raise ValueError('Drive 服务器文件摘要不匹配或缺失')


class DriveStore:
    def __init__(self, root_id, credential_file, quota_project):
        from google.auth.transport.requests import AuthorizedSession
        from google.oauth2.credentials import Credentials
        credential = Credentials.from_authorized_user_file(credential_file)
        self.http = AuthorizedSession(credential.with_quota_project(quota_project))
        self.root_id = root_id
        self.folders = {}

    def request(self, method, url, **kwargs):
        for attempt in range(6):
            try:
                response = self.http.request(method, url, timeout=(20, 180), **kwargs)
                if response.status_code < 400:
                    return response
                try:
                    reasons = [item.get('reason', 'unknown') for item in response.json().get('error', {}).get('errors', [])]
                except (ValueError, AttributeError, TypeError):
                    reasons = []
                retryable = response.status_code in (408, 429, 500, 502, 503, 504) or (
                    response.status_code == 403 and bool(set(reasons) & {'rateLimitExceeded', 'userRateLimitExceeded', 'backendError'}))
                logger.warning('Drive 请求未完成：操作=%s HTTP=%d 原因=%s 尝试=%d',
                               method, response.status_code, reasons, attempt + 1)
                if not retryable:
                    raise DriveRequestError(response.status_code, reasons)
            except requests.RequestException as error:
                logger.warning('Drive 网络暂时失败：操作=%s，重试=%d，类型=%s', method, attempt + 1, type(error).__name__)
            time.sleep(min(30, 2**attempt))
        raise RuntimeError('Drive 请求多次重试仍未完成')

    def child(self, parent, name):
        escaped = name.replace('\\', '\\\\').replace("'", "\\'")
        result = self.request('GET', 'https://www.googleapis.com/drive/v3/files', params={
            'q': f"'{parent}' in parents and name = '{escaped}' and trashed = false",
            'fields': 'files(id,name,mimeType)', 'pageSize': 100}).json()['files']
        if len(result) > 1:
            raise ValueError(f'Drive 存在同名文件，拒绝猜测：{name}')
        return result[0]['id'] if result else None

    def folder(self, parent, name):
        key = (parent, name)
        if key not in self.folders:
            self.folders[key] = self.child(parent, name) or self.request(
                'POST', 'https://www.googleapis.com/drive/v3/files',
                json={'name': name, 'parents': [parent], 'mimeType': 'application/vnd.google-apps.folder'},
                params={'fields': 'id'}).json()['id']
        return self.folders[key]

    def metadata(self, file_id):
        return self.request('GET', 'https://www.googleapis.com/drive/v3/files/' + file_id,
                            params={'fields': FIELDS}).json()

    def _uploaded_object(self, file_id, expected):
        """最终响应可能丢失，只以同一文件 ID 的服务器摘要确认成功。"""
        try:
            check_remote(self.metadata(file_id), expected)
        except DriveRequestError as error:
            if error.status != 404:
                raise
            return None
        except ValueError:
            return None
        return dict(expected, id=file_id)

    def upload(self, path, parent, name=None):
        path = Path(path)
        name = name or path.name
        expected = fingerprints(path)
        existing = self.child(parent, name)
        if existing:
            verified = self._uploaded_object(existing, expected)
            if verified:
                return verified
            logger.info('更新 Drive 文件：%s', name)
        # 预先分配 ID，失效会话重建时继续使用同一对象，避免生成重名副本。
        file_id = existing or self.request('GET', 'https://www.googleapis.com/drive/v3/files/generateIds',
                                          params={'count': 1, 'space': 'drive'}).json()['ids'][0]
        total = expected['bytes']
        for session_attempt in range(3):
            if session_attempt:
                verified = self._uploaded_object(file_id, expected)
                if verified:
                    logger.info('Drive 响应丢失后确认文件完整：文件=%s id=%s', name, file_id)
                    return verified
                try:
                    self.metadata(file_id)
                    existing = file_id
                except DriveRequestError as error:
                    if error.status != 404:
                        raise
                    existing = None
                time.sleep(min(10, 2 ** session_attempt))
            body = {'name': name} if existing else {'id': file_id, 'name': name, 'parents': [parent]}
            url = 'https://www.googleapis.com/upload/drive/v3/files' + ('/' + file_id if existing else '')
            response = self.request('PATCH' if existing else 'POST', url, json=body,
                params={'uploadType': 'resumable', 'fields': FIELDS},
                headers={'X-Upload-Content-Type': 'application/octet-stream', 'X-Upload-Content-Length': str(total)})
            location = response.headers['Location']
            offset, stalled = 0, 0
            logger.info('Drive 直接上传开始：文件=%s 字节=%d 会话尝试=%d id=%s', name, total, session_attempt + 1, file_id)
            try:
                with path.open('rb') as stream:
                    while True:
                        stream.seek(offset)
                        block = stream.read(64 * 1024**2)
                        try:
                            response = self.http.put(location, data=block, headers={
                                'Content-Range': f'bytes {offset}-{offset + len(block) - 1}/{total}' if block else f'bytes */{total}'},
                                timeout=(20, 180))
                        except requests.RequestException:
                            # 请求可能已被接收，先查询服务器偏移，不能盲目重发。
                            response = self.request('PUT', location, data=b'', headers={'Content-Range': f'bytes */{total}'})
                        if response.status_code not in (200, 201, 308):
                            if response.status_code in (408, 429, 500, 502, 503, 504):
                                time.sleep(5)
                                response = self.request('PUT', location, data=b'', headers={'Content-Range': f'bytes */{total}'})
                            if response.status_code not in (200, 201, 308):
                                try:
                                    reasons = [item.get('reason', 'unknown') for item in response.json().get('error', {}).get('errors', [])]
                                except (ValueError, AttributeError, TypeError):
                                    reasons = []
                                raise DriveRequestError(response.status_code, reasons)
                        if response.status_code in (200, 201):
                            if response.json()['id'] != file_id:
                                raise ValueError('Drive 返回的文件 ID 与预分配对象不一致')
                            check_remote(self.metadata(file_id), expected)
                            logger.info('Drive 服务器摘要校验通过：文件=%s 字节=%d id=%s', name, total, file_id)
                            return dict(expected, id=file_id)
                        received = int(response.headers['Range'].rsplit('-', 1)[-1]) + 1 if response.headers.get('Range') else 0
                        if not 0 <= received <= total:
                            raise ValueError('Drive 上传偏移超出文件范围')
                        stalled = stalled + 1 if received <= offset else 0
                        if stalled >= 6:
                            raise RuntimeError('Drive 上传连续六次没有取得进展')
                        offset = received
                        logger.info('Drive 上传进度：文件=%s 已接收=%d/%d', name, offset, total)
            except DriveRequestError as error:
                # 404/410 表示会话失效；权限和容量问题仍然直接失败，不能无限消耗 GPU。
                if error.status not in (404, 410):
                    raise
                logger.warning('Drive 上传会话失效：文件=%s HTTP=%d 原因=%s，将核对对象后有限重建',
                               name, error.status, error.reasons)
        verified = self._uploaded_object(file_id, expected)
        if verified:
            return verified
        raise RuntimeError(f'Drive 上传会话重建三次仍失败：{name}；原完整备份保留')

    def json_file(self, value, parent, name, workspace):
        path = Path(workspace) / ('.drive-' + name)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        try:
            return self.upload(path, parent, name)
        finally:
            path.unlink(missing_ok=True)

    def publish(self, source, stage, step):
        """全部对象校验后才提交清单和指针；从不删除旧的云端完整副本。"""
        source = Path(source)
        parent = self.folder(self.root_id, stage)
        target = self.folder(parent, f'step_{step:06d}')
        files = {}
        for path in sorted(source.iterdir()):
            if path.name.startswith('.') or path.name == 'BACKUP_COMPLETE.json':
                continue
            if not path.is_file() or path.is_symlink():
                raise ValueError('不支持的断点文件类型')
            files[path.name] = self.upload(path, target)
        manifest = {'step': step, 'files': files, 'drive_folder_id': target,
                    'verification': 'Drive API 服务器大小和内容摘要', 'created_at': time.time()}
        self.json_file(manifest, target, 'BACKUP_COMPLETE.json', source)
        self.verify(manifest)
        self.json_file({'step': step, 'path': f'step_{step:06d}', 'drive_folder_id': target}, parent, 'LATEST.json', source)
        return manifest

    def verify(self, manifest):
        for expected in manifest['files'].values():
            check_remote(self.metadata(expected['id']), expected)


def configured_store():
    root_id = os.environ.get('RV_DRIVE_ROOT_ID')
    if not root_id:
        return None
    return DriveStore(root_id, os.environ['RV_DRIVE_CREDENTIAL_FILE'], os.environ['RV_DRIVE_QUOTA_PROJECT'])
