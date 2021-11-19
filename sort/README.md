### 算法效果

<p align="center">
    <img width="50%" src="https://user-images.githubusercontent.com/30433053/129135245-98644c75-9672-4bbf-832c-be6dc9dfea59.png" style="max-width:35%;">
</p>

### 安装依赖

- filterpy==1.4.5
- scikit-image==0.17.2
- lap==0.4.0


### 使用说明

#### python 

```bashrc
$ wget https://motchallenge.net/sequenceVideos/PETS09-S2L1-raw.mp4 -O data/PETS09-S2L1-raw.mp4
$ ./examples/sort.py
```

#### C++

```bashrc
$ mkdir build && cd build
$ cmake .. && make
$ ./sort
```


### 相关文档

- [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)
- [多目标追踪 SORT 算法：Simple Online and Realtime Tracking
](https://yunyang1994.gitee.io/2021/08/14/多目标追踪SORT算法-Simple-Online-and-Realtime-Tracking/)
