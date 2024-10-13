from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='Cython TaTu',
    ext_modules=cythonize("cytatu.pyx", compiler_directives={'language_level': "3", 'boundscheck': False, 'wraparound': False}),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)