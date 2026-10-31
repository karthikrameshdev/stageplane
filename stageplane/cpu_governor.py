"""Bound average process CPU on bare hosts; Docker/cgroups enforce their own quota."""
import os
import signal
import sys
import time

import psutil


def main(pid, cores):
    process = psutil.Process(pid)
    created = process.create_time()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    def cpu_time():
        value = process.cpu_times()
        return value.user + value.system
    try:
        while process.is_running() and process.create_time() == created:
            tick, used = time.monotonic(), cpu_time()
            time.sleep(.02)
            debt = (cpu_time() - used) / cores - (time.monotonic() - tick)
            if debt > .01:
                os.kill(pid, signal.SIGSTOP)
                until = time.monotonic() + debt
                while time.monotonic() < until and process.is_running():
                    time.sleep(min(.1, max(0, until - time.monotonic())))
                os.kill(pid, signal.SIGCONT)
    except (psutil.NoSuchProcess, ProcessLookupError):
        pass
    finally:
        if process.is_running() and process.create_time() == created:
            os.kill(pid, signal.SIGCONT)


if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]))
