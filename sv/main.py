import torch
from torch import nn
from torch.nn import functional as F
import torchaudio
import sys

from models.ecapa_tdnn import ECAPA_TDNN, ECAPA_TDNN_SMALL
from dataset import VoxCeleb1HTestDataset, KathbathUrduDataset, DataLoader
from finetune import train_model
from utils import calculate_eer
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask and attn_mask is deprecated.")

# paths to the pre-trained weights
w2v_ckpt = 'ckpts/wav2vec2_xlsr_SV_fixed.th'
hubert_ckpt = 'ckpts/HuBERT_large_SV_fixed.th'
wavlm_ckpt = 'ckpts/wavlm_large_nofinetune.pth'

# wav2vec_model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr')
hubert_model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='hubert_large_ll60k')
wavlm_model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large')

models = {
    # 'wave2vec_xlsr': (wav2vec_model, w2v_ckpt),
    'hubert_large': (hubert_model, hubert_ckpt),
    'wavlm_large': (wavlm_model, wavlm_ckpt)
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, test_loader):
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(test_loader, total=len(test_loader))
        for batch in pbar:
            wav1, wav2, label = batch
            wav1, wav2, label = wav1.to(device), wav2.to(device), label.to(device)
            wav1 = wav1.squeeze(1).float()
            wav2 = wav2.squeeze(1).float()
            emb1 = model(wav1)
            emb2 = model(wav2)
            sim = F.cosine_similarity(emb1, emb2)
            all_scores.extend(sim.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            running_eer = calculate_eer(all_scores, all_labels)
            pbar.set_postfix(EER=f"{running_eer:.2f}%")

    return all_scores, all_labels

if __name__ == '__main__':
    model = sys.argv[1]
    mode = sys.argv[2] # train or test

    if model not in models:
        print("Select from: wave2vec_xlsr, hubert_large, wavlm_large.")
        sys.exit(1)

    selected_model, ckpt = models[model]
    state_dict = torch.load(ckpt, map_location=lambda storage, loc: storage)
    selected_model.load_state_dict(state_dict['model'], strict=False)
    selected_model = selected_model.to(device)

    if mode == 'train':
        fine_ckpt_path = 'ckpts/{model}_finetuned.pth'
        val_dataset = KathbathUrduDataset(mode='valid')
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        train_model(selected_model, val_loader, fine_ckpt_path, device)

    elif mode == 'test':
        # test_dataset = VoxCeleb1HTestDataset() # uncomment for vox dataset
        test_dataset = KathbathUrduDataset(mode='test_known')
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        all_scores, all_labels = evaluate_model(selected_model, test_loader)
        eer = calculate_eer(all_scores, all_labels)
        print(f"EER {mode}: {eer:.2f}%")