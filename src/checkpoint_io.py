import ctypes
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

if os.name == 'nt':
    import msvcrt
    from ctypes import wintypes

    KERNEL = ctypes.WinDLL('kernel32', use_last_error=True)
    KERNEL.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    KERNEL.CreateFileW.restype = wintypes.HANDLE
    KERNEL.CloseHandle.argtypes = [wintypes.HANDLE]


def sharing_retry(operation, timeout=3.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return operation()
        except PermissionError as error:
            if os.name != 'nt' or error.winerror not in (5, 32, 33) or time.monotonic() >= deadline:
                raise
            time.sleep(.025)


def open_shared_read(path):
    if os.name != 'nt':
        return Path(path).open('r', encoding='utf-8-sig')
    handle = KERNEL.CreateFileW(str(Path(path).resolve()), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except OSError:
        KERNEL.CloseHandle(handle)
        raise
    return os.fdopen(descriptor, 'r', encoding='utf-8-sig')


def read_json(path):
    with sharing_retry(lambda: open_shared_read(path)) as stream:
        content = stream.read()
    return json.loads(content)


def atomic_write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    sharing_retry(lambda: os.replace(temporary, path))


def pause_after_checkpoint(path):
    request = os.environ.get('ENDOSAE_PAUSE_FILE')
    if not request or not Path(request).exists():
        return
    path = Path(path)
    for asset in path.parent.iterdir():
        if asset.is_file() and asset.suffix in ('.npy', '.npz', '.jsonl'):
            with asset.open('r+b') as stream:
                os.fsync(stream.fileno())
    atomic_write_json(Path(request).parent / 'pause_checkpoint.json', dict(
        status='CHECKPOINT_SAVED', pid=os.getpid(), checkpoint=str(path.resolve()),
        checkpoint_exists=path.exists(), saved_at=datetime.now().astimezone().isoformat()))
    print('PAUSE_CHECKPOINT_SAVED', path, flush=True)
    raise SystemExit(75)
