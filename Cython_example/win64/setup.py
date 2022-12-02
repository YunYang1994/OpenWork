'''
Copyright 2022 YunYang1994 All Rights Reserved. 
Author: YunYang1994
FilePath: setup.py
Date: 2022-12-02 12:41:12
'''

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from shutil import copyfile
import numpy as np
import site
import os
import shutil
import glob


setup(
	name = 'Pyimage',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("Pyimage",
                 sources=['src/image_module.pyx'],  
                 language='c++',
                 include_dirs=[np.get_include(), "src/"],
                 extra_compile_args=["-std=c++11", "-w", "-O3"],
                 library_dirs=["build/Release"],
                 libraries=['image'],
                 )],
)

for pyd_file in glob.glob("*.pyd"):
    dst_file = "bin/Release/" + pyd_file 
    if os.path.exists(dst_file):
        os.remove(dst_file)
    shutil.move(pyd_file, os.path.dirname(dst_file))
# shutil.rmtree("build")

