# File used to compile Cython functions
import os
import sys
import shutil
import tempfile
from pathlib import Path

from setuptools import setup
from Cython.Build import cythonize

base_dir = Path(__file__).resolve().parent
build_root = Path(tempfile.gettempdir()) / 'dollar_bars_build'
build_root.mkdir(parents=True, exist_ok=True)

os.chdir(base_dir)

source_path = base_dir / 'cython_loops.pyx'
temp_source_path = build_root / source_path.name
shutil.copy2(source_path, temp_source_path)

if len(sys.argv) == 1:
	sys.argv.extend(['build_ext', '--inplace', '--compiler=mingw32', '--build-temp', str(build_root / 'temp')])

setup(ext_modules=cythonize(str(temp_source_path)))
