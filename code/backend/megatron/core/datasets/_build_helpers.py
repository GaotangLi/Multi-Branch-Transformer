# Copyright (c) 2026 MBT contributors. BSD-3-Clause.
"""Build the native dataset extension in an explicitly selected scratch cache."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import sysconfig


def load_helpers():
    name = "megatron.core.datasets.helpers_cpp"
    if name in sys.modules:
        return sys.modules[name]
    configured = os.environ.get("MBT_DATASET_HELPER_CACHE")
    if not configured or not Path(configured).is_absolute():
        raise RuntimeError("Set MBT_DATASET_HELPER_CACHE to an absolute scratch-cache directory")
    root = Path(configured).resolve()
    if root.is_relative_to(Path.home().resolve()):
        raise RuntimeError("Dataset build cache must be outside the home filesystem")
    source = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for filename in ("helpers.cpp", "Makefile"):
        digest.update((source / filename).read_bytes())
    digest.update((sysconfig.get_config_var("SOABI") or "").encode())
    digest.update(importlib.metadata.version("pybind11").encode())
    directory = root / digest.hexdigest()[:20]
    directory.mkdir(parents=True, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    target = directory / ("helpers_cpp" + suffix)
    with (directory / "build.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.is_file():
            temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
            try:
                subprocess.run(["make", "-f", str(source / "Makefile"), "-C", str(directory),
                    f"PYTHON={sys.executable}", f"SRC={source / 'helpers.cpp'}",
                    f"OUT={temporary}"], check=True)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    spec = importlib.util.spec_from_file_location(name, target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load dataset extension at {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module
