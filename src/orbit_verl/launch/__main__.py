"""Entry point of ``python -m orbit_verl.launch``; the work happens in ``cli.main``."""
import sys

from orbit_verl.launch.cli import main

if __name__ == "__main__":
    sys.exit(main())
