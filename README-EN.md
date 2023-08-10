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
  * [ ] non-streaming asr onnx inference
  * [ ] speaker recognition onnx inference
  * [ ] model integration, streaming and non-streaming 2-pass model inference with speaker verification

## CER
To be tested...

## Quick Usage
The project is still under development. Please wait for further updates.
