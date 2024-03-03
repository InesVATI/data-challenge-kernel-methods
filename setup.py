from setuptools import setup
from Cython.Build import cythonize
import numpy
# Build a projects while passing .pyx files to Extension constructor
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#basic-setup-py

setup(
    ext_modules=cythonize("src/utils_fast.pyx"),
    include_dirs=[numpy.get_include()]
)