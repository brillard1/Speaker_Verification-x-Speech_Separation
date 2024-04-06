import torch
from torch import nn
from torch.nn import functional as F
import torchaudio
import sys

from models.ecapa_tdnn import ECAPA_TDNN
from dataset import VoxCeleb1HTestDataset, DataLoader
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Define the paths to the pre-trained weights
ecapa_checkpoint_path = 'ckpts/ecapa_tdnn.pth'
hubert_checkpoint_path = 'ckpts/hubert_large.pth'
wavlm_large_checkpoint_path = 'ckpts/wavlm_large.pth'

# Load pre-trained models
ecapa_model = ECAPA_TDNN()
ecapa_model.load_state_dict(torch.load(ecapa_checkpoint_path))

hubert_model = ECAPA_TDNN()
hubert_model.load_state_dict(torch.load(hubert_checkpoint_path))

wavlm_model = ECAPA_TDNN()
wavlm_model.load_state_dict(torch.load(wavlm_large_checkpoint_path))

models = {
    'ecapa_tdnn': ecapa_model,
    'hubert_large': hubert_model,
    'wavlm_large': wavlm_model
}

def calculate_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100

def evaluate_model(model, test_loader):
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            wav1, wav2, label = batch
            emb1 = model(wav1)
            emb2 = model(wav2)
            sim = F.cosine_similarity(emb1, emb2)
            all_scores.extend(sim.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    return all_scores, all_labels

if __name__ == '__main__':
    mode = sys.argv[1]

    if mode not in models:
        print("Select from: ecapa_tdnn, hubert_large, wavlm_large.")
        sys.exit(1)

    selected_model = models[mode]

    # Preparing test dataset
    test_dataset = VoxCeleb1HTestDataset(list_file='../dataset/vox/list_test_hard.txt', root_dir='../dataset/vox/wav')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_scores, all_labels = evaluate_model(selected_model, test_loader)
    eer = calculate_eer(all_scores, all_labels)
    print(f"EER using {mode}: {eer:.2f}%")