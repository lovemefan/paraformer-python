# -*- coding:utf-8 -*-
# @FileName  :test_speaker_verification.py
# @Time      :2023/8/12 16:28
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com

from paraformerOnline.runtime.python.svInfer import SpeakerVerificationInfer


def test():
    audio_file1 = "test/a_cn_16k.wav"
    audio_file2 = "test/b_cn_16k.wav"
    audio_file3 = "test/c_cn_16k.wav"

    # threshold 越高区分度越高，返回的说话人身份越多
    model = SpeakerVerificationInfer(model_name="cam++", threshold=0.9)
    index = model.recognize(audio_file1)
    print(index)
    index = model.recognize(audio_file1)
    print(index)
    index = model.recognize(audio_file2)
    print(index)
    index = model.recognize(audio_file2)
    print(index)
    index = model.recognize(audio_file3)
    print(index)
    index = model.recognize(audio_file3)
    print(index)


if __name__ == "__main__":
    test()
