#!/usr/bin/env python3
"""Run comprehensive benchmark."""
import sys
from pathlib import Path

# Add kernelweave to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.run_comprehensive import run_comprehensive_benchmark, main

if __name__ == "__main__":
    main()
