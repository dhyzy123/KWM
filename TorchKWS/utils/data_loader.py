"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/3/15 下午4:59
@Author  : Yang "Jan" Xiao 
@Description : data_loader
"""
import os
import random
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import numpy as np
import librosa
from networks.kwm.utils import time_shift, resampling, spec_augment
from torchaudio.transforms import MFCC

class SpeechCommandDataset(Dataset):
    def __init__(self, dataset_path, json_filename, is_training, class_list, class_encoding, noise_aug=False):
        super(SpeechCommandDataset, self).__init__()
        self.classes = class_list
        self.sampling_rate = 16000
        self.sample_length = 16000
        self.dataset_path = dataset_path
        # root is the parent folder of the dataset.
        self.root = os.path.dirname(dataset_path)
        self.json_filename = json_filename
        self.is_training = is_training
        self.class_encoding = class_encoding
        self.noise_aug = noise_aug
        self.noise_path = os.path.join(dataset_path, "_background_noise_")
        self.noise_dataset = self.load_noise_dataset()
        self.speech_dataset = self.load_speech_dataset()
        self.audio_settings = {
            "sr": 16000,
            "n_mels": 40,
            "n_fft": 480,
            "win_length": 480,
            "hop_length": 160,
            "center": False
        }
        self.aug_settings ={
            "resample": {
                "r_min": 0.85,
                "r_max": 1.15
            },
            "time_shift": {
                "s_min": -0.1,
                "s_max": 0.1
            },
            "spec_aug": {
                "n_time_masks": 2,
                "time_mask_width": 25,
                "n_freq_masks": 2,
                "freq_mask_width": 7
            }
        }
        self.mfcc = MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={"n_fft": 480, "win_length": 480, "hop_length": 160, "n_mels": 40, "center": False},
        )

    def load_noise_dataset(self):
        noise_dataset = []
        for root, _, filenames in sorted(os.walk(self.noise_path, followlinks=True)):
            for fn in sorted(filenames):
                name, ext = os.path.splitext(fn)
                if ext == ".wav":
                    noise_dataset.append(os.path.join(root, fn))
        return noise_dataset

    def load_speech_dataset(self):
        with open(self.json_filename, 'r') as f:
            json_data = [json.loads(line) for line in f.readlines()]

        dataset_list = []
        for item in json_data:
            category = item["command"]
            if category in self.classes:
                category_label = category
            else:
                category_label = "unknown"
            dataset_list.append([item["audio_filepath"], category_label])
        return dataset_list
    
    def _spec_augmentation(self, x,
                           num_time_mask=1,
                           num_freq_mask=1,
                           max_time=25,
                           max_freq=25):

        """perform spec augmentation 
        Args:
            x: input feature, T * F 2D
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
        Returns:
            augmented feature
        """
        max_freq_channel, max_frames = x.size()

        # time mask
        for i in range(num_time_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_time)
            end = min(max_frames, start + length)
            x[:, start:end] = 0

        # freq mask
        for i in range(num_freq_mask):
            start = random.randint(0, max_freq_channel - 1)
            length = random.randint(1, max_freq)
            end = min(max_freq_channel, start + length)
            x[start:end, :] = 0

        return x
    
    def load_audio(self, speech_path):
        waveform, _ = torchaudio.load(speech_path)

        if waveform.shape[1] < self.sample_length:
            # padding if the audio length is smaller than samping length.
            waveform = F.pad(waveform, [0, self.sample_length - waveform.shape[1]])
       
        if self.is_training:
            pad_length = int(waveform.shape[1] * 0.1)
            waveform = F.pad(waveform, [pad_length, pad_length])
            offset = torch.randint(0, waveform.shape[1] - self.sample_length + 1, size=(1,)).item()
            waveform = waveform.narrow(1, offset, self.sample_length)

            if self.noise_aug == True and random.random() < 0.8: 
                noise_index = torch.randint(0, len(self.noise_dataset), size = (1,)).item()
                noise, _ = torchaudio.load(self.noise_dataset[noise_index])
                offset = torch.randint(0, noise.shape[1] - self.sample_length + 1, size = (1, )).item()
                noise  = noise.narrow(1, offset, self.sample_length)
                background_volume = torch.rand(size = (1, )).item() * 0.1
                waveform.add_(noise.mul_(background_volume)).clamp(-1, 1) 

        return waveform

    def one_hot(self, speech_category):
        encoding = self.class_encoding[speech_category]
        return encoding

    def __len__(self):
        return len(self.speech_dataset)
    
    def __getitem__(self, index):

        speech_path, speech_category = self.speech_dataset[index]
        speech_path = os.path.join(self.root, speech_path)
        label = self.one_hot(speech_category)

        if speech_category == "silence":
            waveform = torch.zeros(1, self.sampling_rate)
        else:
            waveform = self.load_audio(speech_path)

        x = waveform.squeeze()
        # waveform
        if self.is_training:
            x = time_shift(x, self.audio_settings["sr"], **self.aug_settings["time_shift"])
            x, _ = resampling(x, self.audio_settings["sr"], **self.aug_settings["resample"])
        if not self.is_training:
            x = x.detach().cpu().numpy()
        x = librosa.util.fix_length(data=x, size=self.audio_settings["sr"])
        
        # librosa MFCC
        # x = librosa.feature.melspectrogram(y=x, **self.audio_settings)
        # x = librosa.feature.mfcc(S=librosa.power_to_db(x), n_mfcc=40)
        # x = librosa.feature.mfcc(y=x, n_mfcc=40, **self.audio_settings)
        # augment
        # if self.is_training:
            # x = spec_augment(x, **self.aug_settings["spec_aug"])
        # x = torch.from_numpy(x).float().unsqueeze(0)
        
        # torch MFCC
        x = torch.from_numpy(x).float()
        x = self.mfcc(x)
        # augment
        if self.is_training:
            x = spec_augment(x, **self.aug_settings["spec_aug"])
        x = x.unsqueeze(0)

        # [1,F,T]
        return x, label