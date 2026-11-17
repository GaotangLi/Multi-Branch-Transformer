"""Package the native source backend without compiling CUDA/C++ at install time."""
from setuptools import find_packages, setup

setup(
    packages=find_packages("code", include=["mbt", "mbt.*"]) + ["mbt_backend"],
    package_dir={"": "code", "mbt_backend": "code/backend"},
    package_data={"mbt_backend": [
        "**/*.py", "**/*.cpp", "**/*.h", "**/*.cu", "**/Makefile",
        "LICENSE", "SOURCE_MANIFEST.json", "PORTABILITY.md",
    ]},
)
