import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

from config import *
from dataset import SoilDataset

weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

test_df = pd.read_csv(TEST_CSV)
test_dataset = SoilDataset(test_df, TEST_IMG_DIR, transform=transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

predictions = []

with torch.no_grad():
    for images, image_ids in tqdm(test_loader, desc="Predicting"):
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
        for img_id, pred in zip(image_ids, preds):
            predictions.append({
                'image_id': img_id,
                'soil_type': inv_label_map[pred]
            })

submission = pd.DataFrame(predictions)
submission.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")
