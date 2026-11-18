"""Minimal runner so the tests work without pytest: ``python -m tests``."""

import sys
import time
import traceback

from tests import test_nfbeam as module

failed = 0
for name in sorted(n for n in dir(module) if n.startswith("test_")):
    start = time.perf_counter()
    try:
        getattr(module, name)()
        print(f"PASS {name} ({time.perf_counter() - start:.1f}s)")
    except Exception:
        failed += 1
        print(f"FAIL {name}")
        traceback.print_exc()
sys.exit(1 if failed else 0)
