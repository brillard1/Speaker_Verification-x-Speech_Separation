import os
import torch
import pandas as pd
from speechbrain.inference.separation import SepformerSeparation as separator
from torch.utils.data import DataLoader
from dataset import CustomLibriMixDataset
import torchaudio
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio, SignalDistortionRatio
from tqdm import tqdm
import sys

BATCH_SIZE = 4
NUM_EPOCHS = 10
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
LR = 1e-3

def finetune(model, train_loader, valid_loader):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best_val_loss = float('inf')

    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0
        t_samples = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}', leave=False)
        for mix, wav1, wav2 in pbar:
            optimizer.zero_grad()
            est_sources = model.separate_batch(mix)
            loss = loss_fn(est_sources[:, :, 0], wav1) + loss_fn(est_sources[:, :, 0], wav2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            t_samples += mix.size(0)
            pbar.set_postfix(train_loss=running_loss / t_samples)
            pbar.update(1)

        val_loss, val_sisnri, val_sdr = evaluate(model, valid_loader)
        pbar.set_postfix_str(f"Train Loss: {running_loss / t_samples} Val Loss: {val_loss} Val SISNRi: {val_sisnri} Val SDRi: {val_sdr}")

        if val_loss < best_val_loss:
            ckpt_path = 'ckpt/best.pt'
            torch.save(model.state_dict(), ckpt_path)
            best_val_loss = val_loss

# testing procedure
def evaluate(model, data_loader):
    loss_fn = torch.nn.MSELoss()
    sisnr = ScaleInvariantSignalNoiseRatio()
    sdr = SignalDistortionRatio()
    running_loss = 0.0
    running_sisnri = 0.0
    running_sdr = 0.0
    t_samples = 0

    model.eval()
    with torch.no_grad():
        for mix, wav1, wav2 in data_loader:
            est_sources = model.separate_batch(mix)
            loss = loss_fn(est_sources[:, :, 0], wav1) + loss_fn((est_sources[:, :, 1], wav2))
            running_loss += loss.item()
            # Compute SISNRi and SDRi
            sisnri = sisnr(est_sources, torch.stack((wav1, wav2), dim=1))
            sdr = sdr(est_sources, torch.stack((wav1, wav2), dim=1))
            running_sisnri += sisnri.sum().item()
            running_sdr += sdr.sum().item()
            t_samples += mix.size(0)

    avg_loss = running_loss / t_samples
    avg_sisnri = running_sisnri / t_samples
    avg_sdr = running_sdr / t_samples

    return avg_loss, avg_sisnri, avg_sdr

if __name__ == '__main__':
    mode = sys.argv[1]

    # pre-trained model
    model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr')
    model = model.to(DEVICE)

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
