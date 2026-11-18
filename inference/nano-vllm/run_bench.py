# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Run the serving benchmark (benchmarks/mbt/runtime.py) on one export with this checkout, never a sibling nano-vLLM."""
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
BENCH = HERE / 'benchmarks/mbt'


def main():
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(HERE), str(BENCH)]
    import nanovllm
    import closed_loop
    for module, root in ((nanovllm, HERE), (closed_loop, BENCH)):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError('Inference import escaped the repository: ' + module.__file__)
    runpy.run_path(str(BENCH/'runtime.py'), run_name='__main__')


if __name__ == '__main__':
    main()
