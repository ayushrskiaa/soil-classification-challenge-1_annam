import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import SoilDataset
from config import *

weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

train_df = pd.read_csv(TRAIN_CSV)
full_train_dataset = SoilDataset(train_df, TRAIN_IMG_DIR, transform=transform)

val_size = int(0.2 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.numpy())

    val_f1 = f1_score(val_labels, val_preds, average='macro')
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f} | Validation F1 Score: {val_f1:.4f}")

torch.save(model.state_dict(), "model.pth")
