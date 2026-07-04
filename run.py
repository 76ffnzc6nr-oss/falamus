#!/usr/bin/env python3
"""falamus entry point (dev convenience).

After `pip install -e .` you can just run the `falamus` command instead. Running this script
directly also works: Python puts the repo root on sys.path, so `import falamus` resolves.

Usage:
    python run.py [workdir] [--resume <sid>] [--exec "<task>"]
    falamus       [workdir] [--resume <sid>] [--exec "<task>"]   (after pip install)
"""

from falamus.main import main

if __name__ == "__main__":
    main()
