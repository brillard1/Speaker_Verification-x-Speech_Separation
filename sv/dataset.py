import os
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import torchaudio
import json

class VoxCeleb1HTestDataset(Dataset):
    def __init__(self, list_file='../dataset/vox/list_test_hard.txt', root_dir='../dataset/vox/wav', target_duration_sec=7, sampling_rate=16000):
        self.labels = []
        self.audio_files = []
        self.target_length = target_duration_sec * sampling_rate

        with open(list_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                label, audio1_path, audio2_path = line.strip().split(' ')
                label = int(label)
                audio1_path = os.path.join(root_dir, audio1_path)
                audio2_path = os.path.join(root_dir, audio2_path)

                if os.path.exists(audio1_path) and os.path.exists(audio2_path):
                    self.audio_files.append((audio1_path, audio2_path))
                    self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio1_path, audio2_path = self.audio_files[idx]
        label = self.labels[idx]

        wav1, sr1 = torchaudio.load(audio1_path)
        wav2, sr2 = torchaudio.load(audio2_path)

        # Pad or truncate the waveforms to a fixed length
        wav1 = self._process_waveform(wav1)
        wav2 = self._process_waveform(wav2)

        return wav1, wav2, label

    def _process_waveform(self, waveform):
        waveform_length = waveform.shape[-1]
        if waveform_length < self.target_length:
            padding = torch.zeros((1, self.target_length - waveform_length))
            waveform = torch.cat((waveform, padding), dim=-1)
        elif waveform_length > self.target_length:
            waveform = waveform[..., :self.target_length]
        return waveform

class KathbathUrduDataset(Dataset):
    def __init__(self, mode, target_duration_sec=7, sampling_rate=16000):
        self.labels = []
        self.audio_files = []
        self.target_length = target_duration_sec * sampling_rate

        metadata_file=f'../dataset/kathbath/{mode}_data.txt'
        root_dir=f'../dataset/kathbath/kb_data_clean_wav/urdu/{mode}/audio'

        with open(metadata_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                label, audio1_path, audio2_path = line.strip().split('\t')
                label = int(label)

                audio1_file = os.path.basename(audio1_path)
                audio2_file = os.path.basename(audio2_path)
                audio1_folder = audio1_file.split('-')[1]
                audio2_folder = audio2_file.split('-')[1]

                audio1_path = os.path.join(root_dir, audio1_folder, audio1_file)
                audio2_path = os.path.join(root_dir, audio2_folder, audio2_file)

                if os.path.exists(audio1_path) and os.path.exists(audio2_path):
                    self.audio_files.append((audio1_path, audio2_path))
                    self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio1_path, audio2_path = self.audio_files[idx]
        label = self.labels[idx]

        wav1, sr1 = torchaudio.load(audio1_path)
        wav2, sr2 = torchaudio.load(audio2_path)

        # Pad or truncate the waveforms to a fixed length
        wav1 = self._process_waveform(wav1)
        wav2 = self._process_waveform(wav2)

        return wav1, wav2, int(label)

    def _process_waveform(self, waveform):
        waveform_length = waveform.shape[-1]
        if waveform_length < self.target_length:
            padding = torch.zeros((1, self.target_length - waveform_length))
            waveform = torch.cat((waveform, padding), dim=-1)
        elif waveform_length > self.target_length:
            waveform = waveform[..., :self.target_length]
        return waveform