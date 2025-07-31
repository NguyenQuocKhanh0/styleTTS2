#coding: utf-8
import os.path as osp
import random
import numpy as np
import random
import soundfile as sf
import librosa

import torch
import torchaudio
import torch.utils.data
import torch.distributed as dist
from multiprocessing import Pool

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import pandas as pd

class TextCleaner:
    def __init__(self, symbol_dict, debug=True):
        self.word_index_dictionary = symbol_dict
        self.debug = debug
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError as e:
                if self.debug:
                    print("\nWARNING UNKNOWN IPA CHARACTERS/LETTERS: ", char)
                    print("To ignore set 'debug' to false in the config")
                continue
        return indexes

np.random.seed(1)
random.seed(1)
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
}

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 root_path,
                 symbol_dict,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 debug=True
                 ):

        _data_list = [l.strip().split('|') for l in data_list]
        self.data_list = _data_list #[data if len(data) == 3 else (*data, 0) for data in _data_list] #append speakerid=0 for all
        self.text_cleaner = TextCleaner(symbol_dict, debug)
        self.sr = sr

        self.df = pd.DataFrame(self.data_list)

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        
        self.root_path = root_path

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):        
        data = self.data_list[idx]
        path = data[0]
        
        wave, text_tensor = self._load_tensor(data)
        
        mel_tensor = preprocess(wave).squeeze()
        
        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]
        
        return acoustic_feature, text_tensor, path, wave

    def _load_tensor(self, data):
        wave_path, text = data
        wave, sr = sf.read(osp.join(self.root_path, wave_path))
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
            print(wave_path, sr)
        
        # Adding half a second padding.
        wave = np.concatenate([np.zeros([12000]), wave, np.zeros([12000])], axis=0) 
        
        text = self.text_cleaner(text)
        
        text.insert(0, 0)
        text.append(0)
        
        text = torch.LongTensor(text)

        return wave, text

    def _load_data(self, data):
        wave, text_tensor = self._load_tensor(data)
        mel_tensor = preprocess(wave).squeeze()

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave
        

    def __call__(self, batch):
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[0].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][0].size(0)
        max_mel_length = max([b[0].shape[1] for b in batch])
        max_text_length = max([b[1].shape[0] for b in batch])

        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        paths = ['' for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]
        
        for bid, (mel, text, path, wave) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            
            waves[bid] = wave

        return waves, texts, input_lengths, mels, output_lengths


def get_length(wave_path, root_path):
    info = sf.info(osp.join(root_path, wave_path))
    return info.frames * (24000 / info.samplerate)

def build_dataloader(path_list,
                     root_path,
                     symbol_dict,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):
    
    dataset = FilePathDataset(path_list, root_path, symbol_dict, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)
    
    print("Getting sample lengths...")
    
    num_processes = num_workers * 2
    if num_processes != 0:
        list_of_tuples = [(d[0], root_path) for d in dataset.data_list]
        with Pool(processes=num_processes) as pool:
            sample_lengths = pool.starmap(get_length, list_of_tuples, chunksize=16)
    else:
        sample_lengths = []
        for d in dataset.data_list:
            sample_lengths.append(get_length(d[0], root_path))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=BatchSampler(
            sample_lengths,
            batch_size,
            shuffle=(not validation),
            drop_last=(not validation),
            num_replicas=1,
            rank=0,
        ),
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )

    return data_loader

#https://github.com/duerig/StyleTTS2/
class BatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        sample_lengths,
        batch_sizes,
        num_replicas=None,
        rank=None,
        shuffle=True,
        drop_last=False,
    ):
        self.batch_sizes = batch_sizes
        if num_replicas is None:
            self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank()
        else:
            self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.time_bins = {}
        self.epoch = 0
        self.total_len = 0
        self.last_bin = None

        for i in range(len(sample_lengths)):
            bin_num = self.get_time_bin(sample_lengths[i])
            if bin_num != -1:
                if bin_num not in self.time_bins:
                    self.time_bins[bin_num] = []
                self.time_bins[bin_num].append(i)

        for key in self.time_bins.keys():
            val = self.time_bins[key]
            total_batch = self.batch_sizes * num_replicas
            self.total_len += len(val) // total_batch
            if not self.drop_last and len(val) % total_batch != 0:
                self.total_len += 1

    def __iter__(self):
        sampler_order = list(self.time_bins.keys())
        sampler_indices = []

        if self.shuffle:
            sampler_indices = torch.randperm(len(sampler_order)).tolist()
        else:
            sampler_indices = list(range(len(sampler_order)))

        for index in sampler_indices:
            key = sampler_order[index]
            current_bin = self.time_bins[key]
            dist = torch.utils.data.distributed.DistributedSampler(
                current_bin,
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
            )
            dist.set_epoch(self.epoch)
            sampler = torch.utils.data.sampler.BatchSampler(
                dist, self.batch_sizes, self.drop_last
            )
            for item_list in sampler:
                self.last_bin = key
                yield [current_bin[i] for i in item_list]

    def __len__(self):
        return self.total_len

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_time_bin(self, sample_count):
        result = -1
        frames = sample_count // 300
        if frames >= 20:
            result = (frames - 20) // 20
        return result