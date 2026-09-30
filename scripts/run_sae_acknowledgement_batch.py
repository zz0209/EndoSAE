import argparse
import ctypes
import hashlib
import json
import msvcrt
import os
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint_io import atomic_write_json, read_json


def check_resource_permissions(plan):
    resources = sorted({resource for stage in plan['stages'] for resource in stage.get('resources', [])})
    if not resources:
        return
    manager = ROOT.parents[1] / '.resource_manager'
    read_json(manager / 'resources.json')
    for directory in (manager / 'leases', manager / 'archive'):
        scratch = Path(tempfile.mkdtemp(prefix='.permission-check-', dir=directory))
        path = scratch / 'check.json'
        try:
            atomic_write_json(path, {'revision': 1})
            atomic_write_json(path, {'revision': 2})
            assert read_json(path) == {'revision': 2}
        finally:
            path.unlink(missing_ok=True)
            scratch.rmdir()
    print('RESOURCE_PERMISSIONS_OK', ', '.join(resources), flush=True)


def alive(pid):
    if not pid:
        return False
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return None if ctypes.get_last_error() != 87 else False
    code = wintypes.DWORD()
    success = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
    kernel.CloseHandle(handle)
    return code.value == 259 if success else None


def snapshot(run):
    plan = read_json(run / 'batch_plan.json')
    state = read_json(run / 'execution_progress.json') if (run / 'execution_progress.json').exists() else {'status': 'READY'}
    if state['status'] == 'RUNNING' and alive(state.get('pid')) is False:
        state['status'] = 'INTERRUPTED'
    now = time.time()
    rows = []
    total_remaining = 0.
    calibrated = {}
    for item in plan['stages']:
        receipt = run / 'receipts' / (item['id'] + '.json')
        if receipt.exists():
            saved = read_json(receipt)
            calibrated.setdefault(item['kind'], []).append(saved['elapsed_seconds'])
    running = state['status'] in ('READY', 'RUNNING', 'COMPLETE')
    for item in plan['stages']:
        receipt = run / 'receipts' / (item['id'] + '.json')
        progress_path = ROOT / item['progress']
        progress = read_json(progress_path) if progress_path.exists() else {}
        done = receipt.exists()
        active = item['id'] == state.get('stage') and not done
        if item.get('method') and progress.get('method') != item['method']:
            progress = {}
        count = item['units'] if done else progress.get('completed_jobs', progress.get('completed', 0)) if active else 0
        total = progress.get('total_jobs', progress.get('total', item['units'])) if active else item['units']
        elapsed = max(0., now - state.get('stage_started', now)) if active else 0.
        estimate = item.get('estimate_seconds')
        history = calibrated.get(item['kind'], [])
        if history:
            estimate = sum(history) / len(history)
        if active and count > 0 and elapsed > 10:
            initial = state.get('initial_count', 0)
            if count > initial:
                estimate = elapsed * (total - count) / (count - initial)
            else:
                estimate = max(0., estimate - elapsed) if estimate else None
        elif active and estimate:
            estimate = max(0., estimate - elapsed)
        if done:
            estimate = 0.
        if estimate is None or total_remaining is None:
            total_remaining = None
        else:
            total_remaining += estimate
        detail = progress.get('stage', progress.get('phase', progress.get('video', '')))
        if active and 'step' in progress and 'steps' in progress:
            detail += f" · {progress['step']} / {progress['steps']} steps"
        label = item['label'] + (' · ' + detail if active and 'step' in progress else '')
        rows.append(dict(id=item['id'], label=label, kind=item['kind'], completed=count, total=total,
                         unit=item['unit'], status='COMPLETE' if done else state['status'] if active else 'QUEUED',
                         active=active, detail=detail,
                         eta_seconds=0 if done else total_remaining if running else None,
                         eta_at=datetime.fromtimestamp(now + total_remaining).astimezone().isoformat()
                         if running and total_remaining is not None and not done else None))
    return dict(time=datetime.now().astimezone().isoformat(), state=state, rows=rows,
                completed=sum(r['status'] == 'COMPLETE' for r in rows), total=len(rows),
                eta_seconds=total_remaining if running else None, question=plan['question'])


def serve(run, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/api/status':
                payload = json.dumps(snapshot(run), ensure_ascii=False).encode('utf-8')
                mime = 'application/json; charset=utf-8'
            elif self.path in ('/', '/index.html'):
                payload = (ROOT / 'scripts/sae_acknowledgement_monitor.html').read_bytes()
                mime = 'text/html; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    atomic_write_json(run / 'monitor.json', dict(pid=os.getpid(), port=port, url=f'http://127.0.0.1:{port}/'))
    print(f'MONITOR http://127.0.0.1:{port}/', flush=True)
    server.serve_forever()


def execute(run, resume):
    pause = run / 'pause.request.json'
    lock = (run / 'coordinator.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    if resume:
        pause.unlink(missing_ok=True)
    plan_path = run / 'batch_plan.json'
    signature = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    plan = read_json(plan_path)
    (run / 'logs').mkdir(exist_ok=True)
    (run / 'receipts').mkdir(exist_ok=True)
    status_path = run / 'execution_progress.json'
    state = dict(status='RUNNING', pid=os.getpid(), started=time.time(), plan_sha256=signature)
    try:
        check_resource_permissions(plan)
    except PermissionError as error:
        state.update(status='FAILED', failure_kind='resource_directory_permission',
                     error=str(error), action='Restart this coordinator with permission to write the shared resource manager directories.')
        atomic_write_json(status_path, state)
        raise
    for stage in plan['stages']:
        receipt = run / 'receipts' / (stage['id'] + '.json')
        if receipt.exists():
            assert read_json(receipt)['plan_sha256'] == signature
            continue
        state.update(stage=stage['id'], stage_started=time.time())
        if pause.exists():
            state['status'] = 'PAUSED'
            atomic_write_json(status_path, state)
            return
        progress_path = ROOT / stage['progress']
        prior = read_json(progress_path) if progress_path.exists() else {}
        if stage.get('method') and prior.get('method') != stage['method']:
            prior = {}
        state['initial_count'] = prior.get('completed_jobs', prior.get('completed', 0))
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1',
                           ENDOSAE_PAUSE_FILE=str(pause), PYTHONPATH=os.pathsep.join(str(ROOT / part) for part in
                           ['artifacts/environments/endomind-inference-overlay', 'artifacts/environments/bytetrack-overlay', '.']))
        command = stage['command']
        for resource in reversed(stage.get('resources', [])):
            command = [sys.executable, str(ROOT.parents[1] / '.resource_manager/resource_manager.py'), 'run',
                       '--resource', resource, '--project', 'EndoSAE_EndoFM', '--task', run.name + '-' + stage['id'],
                       '--wait-sec', '28800', '--heartbeat-sec', '20', '--'] + command
        atomic_write_json(status_path, state)
        print('STAGE', stage['id'], stage['label'], flush=True)
        with (run / 'logs' / (stage['id'] + '.log')).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(started=state['stage_started'], command=command)) + '\n')
            stream.flush()
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            atomic_write_json(status_path, state)
            code = child.wait()
        if code == 75 and pause.exists():
            state.update(status='PAUSED', exit_code=code)
            atomic_write_json(status_path, state)
            return
        if code:
            state.update(status='FAILED', exit_code=code, log=str(run / 'logs' / (stage['id'] + '.log')))
            atomic_write_json(status_path, state)
            raise SystemExit(code)
        assert (ROOT / stage['output']).exists(), stage['output']
        atomic_write_json(receipt, dict(plan_sha256=signature, elapsed_seconds=time.time() - state['stage_started'],
                                       completed_at=datetime.now().astimezone().isoformat(), output=stage['output']))
    state.update(status='COMPLETE', completed_at=datetime.now().astimezone().isoformat())
    atomic_write_json(status_path, state)
    print('BATCH_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--check-permissions', action='store_true')
    parser.add_argument('--port', type=int, default=57581)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.check_permissions:
        check_resource_permissions(read_json(run / 'batch_plan.json'))
    elif args.serve:
        serve(run, args.port)
    elif args.pause:
        atomic_write_json(run / 'pause.request.json', dict(requested_at=datetime.now().astimezone().isoformat()))
        print('PAUSE_REQUESTED', flush=True)
    else:
        execute(run, args.resume)


if __name__ == '__main__':
    main()
