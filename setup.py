import numpy as np
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize

def numpy_include():
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include


ext_modules = [
    Extension(
        'reid.evaluation.rank_cylib.rank_cy',
        ['reid/evaluation/rank_cylib/rank_cy.pyx'],
        include_dirs=[numpy_include()],
    )
]
__version__ = '1.0.0'

setup(
    name='DASK',
    version='1.0.0',
    description='DASK: Distribution Rehearsing via Adaptive Style Kernel Learning for Exemplar-Free Lifelong Person Re-Identification',
    author='Kunlun Xu',
    license='MIT, following Zhicheng Sun',
    packages=find_packages(),
    keywords=['Person Re-Identification', 'Lifelong Learning', 'Computer Vision'],
    ext_modules=cythonize(ext_modules)
)