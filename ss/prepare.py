import os
import pandas as pd
import numpy as np
import soundfile as sf
from shutil import copyfile
from itertools import combinations
from tqdm import tqdm

np.random.seed(60)  # for reproducibility

# pad or truncate audio files
def process_audio(audio, dur=15, sr=16000):
    target_length = int(dur * sr)
    curr_length = len(audio)

    if curr_length < target_length:
        padded_audio = np.pad(audio, (0, target_length - curr_length), 'constant')
    elif curr_length >= target_length:
        padded_audio = audio[:target_length]

    return padded_audio

def mix_audio(audio1, audio2, snr=5):
    audio1 = process_audio(audio1)
    audio2 = process_audio(audio2)

    # scaling factor w.r.t 1 on 2 based on SNR
    rms_audio1 = np.sqrt(np.mean(np.square(audio1)))
    rms_audio2 = np.sqrt(np.mean(np.square(audio2)))
    target_rms = rms_audio1 / (10 ** (snr / 20))
    scale_factor = target_rms / rms_audio2
    mixed_audio = audio1 + audio2 * scale_factor
    return audio1, audio2, mixed_audio

def generate_librimix(metadata, root_dir, dataset_dir):
    metadata = pd.read_csv(metadata)
    unique_speakers = metadata['speaker_ID'].unique()  # unique speakers
    speaker_pairs = list(combinations(unique_speakers, 2))

    os.makedirs(os.path.join(dataset_dir, 'mix'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'clean'), exist_ok=True)

    annotations = []
    # Iterate over speaker pairs
    for idx, (speaker1, speaker2) in tqdm(enumerate(speaker_pairs), total=len(speaker_pairs)):
        speaker1_files = metadata[metadata['speaker_ID'] == speaker1]['origin_path'].tolist()
        speaker2_files = metadata[metadata['speaker_ID'] == speaker2]['origin_path'].tolist()
        # random files from each speaker
        np.random.shuffle(speaker1_files)
        np.random.shuffle(speaker2_files)

        dataset_set = 'train' if idx % 10 < 7 else 'test'

        for i in range(min(len(speaker1_files), len(speaker2_files))):
            audio1_path = os.path.join(root_dir, speaker1_files[i])
            audio2_path = os.path.join(root_dir, speaker2_files[i])

            mixed_audio_filename = f'{speaker1}_{speaker2}_{i}.wav'
            mixed_audio_path = os.path.join(dataset_dir, 'mix', mixed_audio_filename)
            clean_audio1_path = os.path.join(dataset_dir, 'clean', f'{speaker1}_{i}.wav')
            clean_audio2_path = os.path.join(dataset_dir, 'clean', f'{speaker2}_{i}.wav')

            audio1, sr1 = sf.read(audio1_path)
            audio2, sr2 = sf.read(audio2_path)

            assert sr1 == sr2, "diff sample rates"
            audio1, audio2, mixed_audio = mix_audio(audio1, audio2)

            sf.write(clean_audio1_path, audio1, samplerate=sr1)
            sf.write(clean_audio2_path, audio2, samplerate=sr2)
            sf.write(mixed_audio_path, mixed_audio, samplerate=sr1)

            annotations.append([mixed_audio_path, clean_audio1_path, clean_audio2_path, dataset_set])

    annotations_df = pd.DataFrame(annotations, columns=['mixed_audio', 'target_audio1', 'target_audio2', 'set'])
    annotations_csv_path = os.path.join(dataset_dir, 'CustomLibriMixed_data.csv')
    annotations_df.to_csv(annotations_csv_path, index=False)

if __name__ == "__main__":
    metadata = '../dataset/LibriSpeech/test-clean.csv'
    root_dir = '../dataset/LibriSpeech/'
    dataset_dir = '../dataset/CustomLibriMixed'

    generate_librimix(metadata, root_dir, dataset_dir)
