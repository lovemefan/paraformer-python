<br/>
<h2 align="center">Paraformer online python</h2>
<br/>

[English readme](README-EN.md)

Paraformer是达摩院语音团队提出的一种高效的非自回归端到端语音识别框架，多个公开数据集上取得SOTA效果，缺点是该模型没有标点符号。
该项目为Paraformer中文通用语音识别模型，采用工业级数万小时的标注音频进行模型训练，保证了模型的通用识别效果。
模型可以被应用于语音输入法、语音导航、智能会议纪要等场景。


本项目为Paraformer封装，基于onnxruntime的推理包，代码主要来自[Funasr](https://github.com/alibaba-damo-academy/FunASR)官方

另外praformer的接口服务在另一个项目[Parafirner-webserver](https://github.com/lovemefan/Paraformer-webserver)中

## 目前的进度
* [2023年8月10日] 
  * [x] vad模型onnx推理
  * [x] 标点模型onnx推理
  * [x] 流式asr onnx推理
  * [x] 非流式asr onnx推理
  * [ ] 说话人识别 onnx推理
  * [ ] 模型整合，流式模型非流式 2pass 带有说话人验证的模型推理


## CER
待测试 ...

## 快速使用

项目还没完善，再等等