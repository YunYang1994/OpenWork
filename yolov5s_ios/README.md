<a href="https://apps.apple.com/app/id1452689527" target="_blank">
<img src="https://user-images.githubusercontent.com/30433053/111970562-5fff6500-8b36-11eb-84f5-fe557964a4f5.jpg" width="1000"></a>

## 内容简介

使用 [**yolov5s.onnx**](https://github.com/YunYang1994/openwork/releases/download/v1.0/yolov5s.onnx) 并量化成 FP16 精度的 [**yolov5s_fp16.onnx.mnn（把它放在resource文件下**](https://github.com/YunYang1994/OpenWork/releases/download/v1.0/yolov5s_fp16.onnx.mnn)，像处理使用的是 OpenCV，模型推理使用的是阿里的 [MNN](https://github.com/alibaba/MNN) 框架。可以直接编译安装使用，在 iphone 手机上运行。

## 安装使用

```bashrc
$ pod install
$ open detection.xcworkspace 
```

## 相关文档
- [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- [语雀文档，MNN 使用说明](https://www.yuque.com/mnn/cn)
