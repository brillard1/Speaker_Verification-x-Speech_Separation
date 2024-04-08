import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchaudio

class CustomLibriMixDataset(Dataset):
    def __init__(self, annotations='../dataset/CustomLibriMixed/CustomLibriMixed_data.csv', root_dir='../dataset/', mode='train'):
        self.annotations = pd.read_csv(annotations)
        self.root_dir = root_dir
        self.mode = mode
        self.resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)

        if self.mode not in ['train', 'test']:
            raise ValueError("must be either train or test")

        self.annotations = self.annotations[self.annotations['set'] == mode]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        mix_path = os.path.join(self.root_dir, self.annotations.iloc[idx]['mixed_audio'])
        wav1_path = os.path.join(self.root_dir, self.annotations.iloc[idx]['target_audio1'])
        wav2_path = os.path.join(self.root_dir, self.annotations.iloc[idx]['target_audio2'])

        mix, _ = torchaudio.load(mix_path)
        mix = self.resampler(mix)

        wav1, _ = torchaudio.load(wav1_path)
        wav1 = self.resampler(wav1)

        wav2, _ = torchaudio.load(wav2_path)
        wav2 = self.resampler(wav2)

        return mix, wav1, wav2