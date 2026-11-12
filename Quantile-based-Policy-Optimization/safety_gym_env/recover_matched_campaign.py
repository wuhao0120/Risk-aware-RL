"""Reconnect this campaign's supervisors when the workspace restart script runs."""
import fcntl
import os
import socket
import subprocess
from pathlib import Path

import matched_campaign as c


def ensure_supervisors():
    if (c.ROOT / 'cancelled_by_user.json').exists():
        return
    host = c.ROOT / 'campaign-host'
    if not host.is_file() or host.read_text().strip() != socket.gethostname():
        return
    with (c.ROOT/'recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (c.ROOT/'hold-restored.json').exists() and all(
            (c.directory(j)/'eval.json').exists() or
            ((c.directory(j)/'train.json').exists() and not c.read(c.directory(j)/'train.json')['ok'])
            for j in c.JOBS
        ):
            return
        daemon = next(Path('/nix/store').glob('*supervisor-4.2.2/bin/supervisord'))
        for service in ('supervisord', 'extra-supervisord', 'fourway-supervisord', 'report-supervisord'):
            ini = c.ROOT/f'{service}.ini'
            try:
                pid = int((c.ROOT/f'{service}.pid').read_text())
                args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError, ValueError):
                args = []
            if os.fsencode(ini) not in args:
                subprocess.run([daemon, '-c', str(ini)], check=True)


if __name__ == '__main__':
    ensure_supervisors()
