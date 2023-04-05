# yolov8_ByteTrack 

一个很随意、简单的yolov8 bytetrack跟踪实现，目的是将yolov8推理和ByteTrack独立分离出来，摆脱屎山代码堆，方便代码修改和加入相应模块测试。


## 依赖

基本上就是yolov8的相关依赖，如果你已经用过yolov8，就用那个conda环境就好了，可能还有一些bytetrack的依赖，使用的过程中看着安装


## 运行

将权重和视频文件路径配置好。直接运行infer.py文件即可


代码没有做太多优化，改起来很方便。


## Tensort实现 yolov8_ByteTrack

1. 运行export-det.py 文件，得到outputs修改后的onnx文件


```python
python export-det.py --weights yolov8n.pt --iou-thres 0.65 --conf-thres 0.25 --topk 100 --opset 11 --sim --input-shape 1 3 640 640 --device cuda:0
```

2. 使用trtexec 将onnx转trt

```python
./trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
```

拷贝yolov8n.engine到tensorrt 实现的C++工程中

[yolov8_ByteTrack_TensorRT](https://github.com/1079863482/yolov8_ByteTrack_TensorRT)


## 参考

[YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT)


[ByteTrack](https://github.com/ifzhang/ByteTrack)



