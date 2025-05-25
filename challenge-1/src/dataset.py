import os
from PIL import Image
from torch.utils.data import Dataset
from config import label_map

class SoilDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_test=False):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]['image_id']
        image_path = os.path.join(self.image_dir, image_id)

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, image_id
        else:
            label = label_map[self.df.iloc[idx]['soil_type']]
            return image, label
