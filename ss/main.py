import os
import torch
import pandas as pd
from speechbrain.inference.separation import SepformerSeparation as separator
from torch.utils.data import DataLoader
from dataset import CustomLibriMixDataset
import torchaudio
from tqdm import tqdm
import sys

from torchmetrics.audio import ScaleInvariantSignalNoiseRatio, SignalDistortionRatio
from torchmetrics.audio import PermutationInvariantTraining
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio, signal_distortion_ratio

BATCH_SIZE = 4
NUM_EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-3

def finetune(model, train_loader, valid_loader):
    loss_fn = PermutationInvariantTraining(signal_distortion_ratio, 
                                    mode="speaker-wise", eval_func="max")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best_val_loss = float('inf')

    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0
        t_samples = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}', leave=False)
        for mix, wav1, wav2 in pbar:
            optimizer.zero_grad()
            mix.requires_grad=True
            preds = model.separate_batch(mix) # [batch, time, spk]
            targets = targets = torch.stack((wav1, wav2), dim=-1)
            loss = loss_fn(preds.transpose(1,2), targets.transpose(1,2)) # [batch, spk, time]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            t_samples += mix.size(0)
            pbar.set_postfix(train_loss=running_loss / t_samples, refresh=True)

        val_loss, val_sisnri, val_sdri = evaluate(model, valid_loader)
        print(f"Train Loss: {running_loss / t_samples} Val Loss: {val_loss} Val SISNRi: {val_sisnri} Val SDRi: {val_sdri}")

        if val_loss < best_val_loss:
            ckpt_path = 'ckpt/best.pt'
            os.makedirs('ckpt', exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            best_val_loss = val_loss

# testing procedure
def evaluate(model, data_loader):
    loss_fn = PermutationInvariantTraining(signal_distortion_ratio, 
                                    mode="speaker-wise", eval_func="max")
    SISNR = ScaleInvariantSignalNoiseRatio()
    SDR = SignalDistortionRatio()
    running_loss = 0.0
    running_sisnri = 0.0
    running_sdri = 0.0
    t_samples = 0.0

    model.eval()
    with torch.no_grad():
        for mix, wav1, wav2 in data_loader:
            preds = model.separate_batch(mix) # [batch, time, spk]
            targets = torch.stack((wav1, wav2), dim=-1)
            loss = loss_fn(preds.transpose(1,2), targets.transpose(1,2)) # [batch, spk, time]

            # Compute SI-SNR improvement
            sisnr = SISNR(preds, targets)
            mix_2 = torch.stack([mix] * 2, dim=-1)
            mix_2 = mix_2.to(targets.device)
            sisnr_baseline = SISNR(
                mix_2, targets
            )
            sisnr_i = sisnr - sisnr_baseline

            # Compute SDR improvement
            sdr = SDR(
                preds.transpose(1,2), targets.transpose(1,2)
            )
            sdr_baseline = SDR(
                mix_2.transpose(1,2), targets.transpose(1,2)
            )
            sdr_i = sdr.mean() - sdr_baseline.mean()

            running_loss += loss.item()
            running_sisnri += sisnr_i.sum().item()
            running_sdri += sdr_i.sum().item()
            t_samples += mix.size(0)

    mean_loss = running_loss / t_samples
    mean_sisnri = running_sisnri / t_samples
    mean_sdr = running_sdri / t_samples

    return mean_loss, mean_sisnri, mean_sdr

if __name__ == '__main__':
    mode = sys.argv[1]
    load = False
    if len(sys.argv) == 3:
        load = sys.argv[2] == '--load'

    # pre-trained model
    model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr')
    model = model.to(DEVICE)

    if load:
        print('Loading model from ckpt/best.pt')
        ckpt = torch.load('ckpt/best.pt')
        model.load_state_dict(ckpt)

    # fine-tuning and testing
    train_dataset = CustomLibriMixDataset(mode='train', device=DEVICE)
    valid_dataset = CustomLibriMixDataset(mode='valid', device=DEVICE)
    test_dataset = CustomLibriMixDataset(mode='test', device=DEVICE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if mode == 'train':
        finetune(model, train_loader, valid_loader)
    test_loss, test_sisnri, test_sdr = evaluate(model, test_loader)
    print(f'Test metrics:\nloss: {test_loss}, SISNRi: {test_sisnri}, SDRi: {test_sdr}')
