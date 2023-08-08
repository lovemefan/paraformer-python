# -*- coding:utf-8 -*-
# @FileName  :audioHelper.py
# @Time      :2023/8/8 20:27
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com

import array
import struct
import time

import numpy as np


class AudioReader:
    """

    read audio from sanic request
    """

    def __init__(self):
        pass

    @staticmethod
    def get_info(self, path: str):
        with open(path, "rb") as f:
            (
                name,
                data_lengths,
                _,
                _,
                _,
                _,
                channels,
                sample_rate,
                bit_rate,
                block_length,
                sample_bit,
                _,
                pcm_length,
            ) = struct.unpack_from("<4sL4s4sLHHLLHH4sL", f.read(44))
            assert sample_rate == 16000, "sample rate must be 16000"
            nframes = pcm_length // (channels * 2)
        return nframes

    @staticmethod
    def read_wav_bytes(data: bytes):
        """
        convert bytes into array of pcm_s16le data
        :param data: PCM format bytes
        :return:
        """

        # header of wav file
        info = data[:44]
        frames = data[44:]
        (
            name,
            data_lengths,
            _,
            _,
            _,
            _,
            channels,
            sample_rate,
            bit_rate,
            block_length,
            sample_bit,
            _,
            pcm_length,
        ) = struct.unpack_from("<4sL4s4sLHHLLHH4sL", info)
        # shortArray each element is 16bit
        data = AudioReader.read_pcm_byte(frames)
        return data, sample_rate

    @staticmethod
    def read_wav_file(audio_path: str):
        with open(audio_path, "rb") as f:
            data = f.read()
        return AudioReader.read_wav_bytes(data)

    @staticmethod
    def read_pcm_byte(data: bytes):
        short_array = array.array("h")
        short_array.frombytes(data)
        data = np.array(short_array, dtype="float16") / (1 << 15)
        return data

    @staticmethod
    def export_pcm_to_wav(
        data: bytes, sample_rate=44100, num_channels=1, sample_width=2
    ):
        pcm_length = len(data)
        block_length = int(num_channels * sample_width)
        bits_rate = num_channels * sample_rate * sample_width * 8
        bits_per_sample = sample_width * 8
        header = struct.pack(
            "<4sL4s4sLHHLLHH4sL",
            b"RIFF",
            36 + pcm_length,
            b"WAVE",
            b"fmt ",
            16,
            1,
            1,
            sample_rate,
            bits_rate,
            block_length,
            bits_per_sample,
            b"data",
            pcm_length,
        )
        return header + data

    @staticmethod
    def export_pcm_to_wav_file(data, file_path=None):
        head = AudioReader.export_pcm_to_wav(data)
        file_path = file_path or f"/tmp/{time.time()}.wav"
        with open(file_path, "wb") as f:
            f.write(head)
            f.write(data)
        return file_path
