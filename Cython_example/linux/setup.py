'''
Copyright 2022 YunYang1994 All Rights Reserved. 
Author: YunYang1994
FilePath: setup.py
Date: 2022-12-01 16:15:20
'''

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np
import site
import os
import shutil
import platform
site_path = site.getsitepackages()[0]
print("Install to site package", site_path)

cstudent = Extension('Pystudent', 
                ['src/student/student_module.pyx'],  
                include_dirs=["src/student/include", np.get_include()],               
                library_dirs=["src/student/lib"],
                libraries=['student'],
                extra_compile_args=['-std=c++11'],
                runtime_library_dirs=["$ORIGIN/student/lib"],
                language="c++"                 
                )

# Compile Python module
setup(
    name='Pystudent',
    version='1.0',
    license="YunYang1994 License",
    description=("Python Binding of Student."),
    ext_modules=cythonize(cstudent, compiler_directives={'language_level': 3} ),

    packages=find_packages('src'),              # 包含__init__.py的文件夹）-------和setup.py同一目录下搜索各个含有 init.py 的包(必须包含__init__.py，否则不会打包)
    package_dir={'': 'src'},                    # 哪些目录下的文件被映射到哪个源码包。一个例子：package_dir = {'': 'lib'}，表示“root package”中的模块都在lib 目录中
    package_data={'student':["lib/*.so"]},      # 需要打包到一些数据 
    
    install_requires = ["cython", "numpy"],
    platforms="linux_x86_64"
)





