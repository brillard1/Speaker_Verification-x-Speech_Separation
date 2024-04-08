import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from utils import calculate_eer
from tqdm import tqdm

def train_model(model, val_loader, ckpt_path, device, num_epochs=10, learning_rate=1e-2):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    all_scores = []
    all_labels = []
    best_val_loss = float('inf')
    
    model.train()
    for epoch in range(num_epochs):
        running_val_loss = 0.0
        pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        for batch in pbar:
            optimizer.zero_grad()
            wav1, wav2, label = batch
            wav1, wav2, label = wav1.to(device), wav2.to(device), label.to(device)
            wav1 = wav1.squeeze(1).float()
            wav2 = wav2.squeeze(1).float()
            emb1 = model(wav1)
            emb2 = model(wav2)
            sim = F.cosine_similarity(emb1, emb2)
            # (-1,1) > (0,1)
            sim = (sim + 1) / 2
            loss = criterion(sim.float(), label.float())
            loss.backward()
            optimizer.step()
            running_val_loss += loss.item()

            all_scores.extend(sim.cpu().detach().numpy())
            all_labels.extend(label.cpu().detach().numpy())
            running_eer = calculate_eer(all_scores, all_labels)
            pbar.set_postfix({'val_loss': f'{running_val_loss / (pbar.n + 1):.2f}', 'running_eer': f'{running_eer:.2f}'})
            pbar.update()

        val_loss = running_val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Running EER: {running_eer:.2f}")

        # saving best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)