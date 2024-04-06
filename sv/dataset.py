import os
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import torchaudio

class VoxCeleb1HTestDataset(Dataset):
    def __init__(self, list_file, root_dir, target_duration_sec=7, sampling_rate=16000):
        self.labels = []
        self.audio_files = []
        self.root_dir = root_dir
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

        # wav1 = torch.from_numpy(wav1).unsqueeze(0).float()
        # wav2 = torch.from_numpy(wav2).unsqueeze(0).float()
        # Resample1 = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=16000)
        # Resample2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=16000)
        # wav1 = Resample1(wav1)
        # wav2 = Resample2(wav2)

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