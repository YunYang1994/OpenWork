<!--
 * Copyright 2022 YunYang1994 All Rights Reserved. 
 * @Author: YunYang1994
 * @FilePath: README.md
 * @Date: 2022-12-02 10:23:18
-->

由于动态库路径的问题，Windows 下不适合编译成 wheel 文件，推荐打包成 pyd 文件。首先用 VSCode 编译，然后运行：

```
$ ./bin/Release/demo.exe
$ python setup.py build_ext -i
$ python test.py
```

- [unsigned char 和 uint8_t 有啥区别](https://blog.csdn.net/u011068702/article/details/77917498)
- [use Ctyhon to work with Numpy](https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html)
