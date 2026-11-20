"""Copies stdin to a log file, prefixing every line with the epoch time it arrived at and a tab.

The aggregators read these stamps to split a launch's wall time into start-up and generation.
With --echo the lines also go to stdout unchanged, for a terminal that watches the run.
"""
import sys
import time


def main() -> int:
    path = sys.argv[1]
    echo = "--echo" in sys.argv[2:]
    # Ray forwards worker output byte for byte, and a cut multi-byte character must not stop the log.
    sys.stdin.reconfigure(errors="replace")
    with open(path, "a", encoding="utf-8", errors="replace") as log:
        for line in sys.stdin:
            log.write(f"{time.time():.3f}\t{line}")
            log.flush()
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
