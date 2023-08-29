# Paraformer online python

Paraformer is an efficient non-autoregressive end-to-end speech recognition framework proposed by the Damo Academy Speech Team. It achieves state-of-the-art results on multiple public datasets. One drawback of this model is that it doesn't include punctuation.
This project provides a Chinese general-purpose speech recognition model based on Paraformer, trained on tens of thousands of hours of annotated audio data, ensuring its versatility in recognition performance.
The model can be applied in scenarios such as speech input methods, voice navigation, and intelligent meeting summaries.

This project is an encapsulation of Paraformer and is based on the onnxruntime inference package. The code is mainly derived from the [Funasr](https://github.com/alibaba-damo-academy/FunASR) official repository.

Furthermore, the interface service for Paraformer is available in another project called [Paraformer-webserver](https://github.com/lovemefan/Paraformer-webserver).

## Current Progress
* [August 10, 2023] 
  * [x] vad model onnx inference
  * [x] punctuation model onnx inference
  * [x] streaming asr onnx inference
  * [x] non-streaming asr onnx inference
  * [x] speaker recognition onnx inference
  * [x] model integration, streaming and non-streaming 2-pass model inference with speaker verification

## CER

| 测试集                              | 领域             | paraformer | bilibili | 思必驰 | 阿里  | 百度   | 讯飞  | 微软  | 腾讯  | 依图 |
| :---------------------------------- | ---------------- | --- | -------- | ------ | ----- | ------ | ----- | ----- | ----- | ---- |
| 直播带货 李佳琪薇娅 （770条, 0.9H） | 电商、美妆       | 6.28 | 6.45⬆️    | 10.04⬆️ | 4.33⬇️ | 16.69⬇️ | 9.10⬇️ | 5.29⬇️ | 6.56⬆️ | 7.33 |
| 新闻联播 （5069条, 9H）             | 时政             | 0.6 | 0.57⬇️    | 0.98⬇️  | 0.32⬇️ | 1.56   | 0.81⬇️ | 0.25⬇️ | 1.02⬇️ | 0.76 |
| 访谈 鲁豫有约 （2993条, 3H）        | 工作、说话       | 3.57 | 2.81⬇️    | 3.3⬆️   | 2.29⬇️ | 5.86   | 3.39⬇️ | 2.74⬇️ | 3.51⬆️ | 2.94 |
| 场馆演讲罗振宇跨年 （1311条, 2.7H） | 社会、人文、商业 | 1.98 | 1.57⬇️    | 1.72⬇️  | 1.17⬇️ | 3.23   | 2.18⬆️ | 1.16⬆️ | 1.75⬆️ | 1.49 |
| 在线教育 李永乐 （3148条, 4.4H）    | 科普             | 2.61 | 1.44⬇️    | 2.2⬆️   | 1.0⬇️  | 6.90   | 2.03⬇️ | 1.31⬇️ | 1.78⬇️ | 1.81 |
| 播客 创业内幕 （2251条, 4.2H）      | 创业、产品、投资 | 4.72 | 3.22⬇️    | 4.24⬇️  | 2.43⬇️ | 7.28⬇️  | 3.82⬇️ | 3.61⬇️ | 3.78⬇️ | 3.7  |
| 线下培训 老罗语录 （884条,1.3H）    | 段子、做人       | 4.64 | 3.81⬆️    | 6.46⬆️  | 3.30⬇️ | 14.13⬇️ | 5.66⬇️ | 3.98⬇️ | 5.50⬇️ | 4.76 |
| 直播 王者荣耀 （1561条, 1.6H）      | 游戏             | 6.69 | 5.69⬇️    | 8.14⬆️  | 4.01⬇️ | 10.32⬇️ | 8.31⬆️ | 5.48⬇️ | 6.14⬆️ | 6.92 |
| 电视节目 天下足球 （1683条, 2.7H）  | 足球             | 1.29 | 0.91⬇️    | 1.54⬇️  | 0.61⬇️ | 5.38   | 1.64⬇️ | 0.88⬇️ | 2.68⬇️ | 0.83 |
| 播客故事FM （3466条, 4.5H）         | 人生故事、见闻   | 3.50 | 3.22⬇️    | 3.82⬆️  | 2.22⬇️ | 5.62⬇️  | 3.72⬇️ | 3.28⬇️ | 3.65⬇️ | 3.67 |
| 罗翔   法考（1053条, 4H）           | 法律 法考        | 2.02 | 1.81⬇️    | 2.86⬇️  | 0.94⬇️ | 5.55   | 2.90⬇️ | 1.19⬇️ | 2.02⬇️ | 1.65 |
| 张雪峰 在线教育考研(1170条, 3.5H)   | 考研 高校报考    | 3.43 | 2.05⬇️    | 3.2⬇️   | 1.38⬇️ | 9.34   | 3.15⬇️ | 2.01⬇️ | 2.71⬆️ | 2.61 |
| 谷阿莫 短视频 影剪(1321条, 2.5H)    | 美食、烹饪       | 3.92 | 3.01⬇️    | 4.02⬇️  | 1.94⬇️ | 7.65   | 3.95⬇️ | 4.22⬇️ | 2.94⬇️ | 2.81 |
| 琼斯爱生活 美食&烹饪(856条, 2H)     | 美食、烹饪       | 4.71 | 3.61⬇️    | 6.29⬇️  | 2.53⬇️ | 13.17  | 4.85⬇️ | 3.07⬇️ | 4.56⬇️ | 3.99 |
| 单田芳 评书白眉大侠(1168条, 2.5H)   | 江湖、武侠       | 5.1 | 4.64⬇️    | 9.22⬇️  | 2.5⬇️  | 15.42  | 9.51⬇️ | 5.47⬇️ | 5.89⬆️ | 5.45 |



## Quick Usage
```bash
git clone https://github.com/lovemefan/paraformer-online-python.git
cd paraformer-online-python && pip install .
python test/test_asr_all_in_one.py
```
