import torch

BATCH_SIZE = 32
NUM_CLASSES = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10

TRAIN_CSV = "data/train_labels.csv"
TEST_CSV = "data/test_ids.csv"
TRAIN_IMG_DIR = "data/train"
TEST_IMG_DIR = "data/test"

label_map = {
    'Alluvial soil': 0,
    'Black Soil': 1,
    'Clay soil': 2,
    'Red soil': 3
}
inv_label_map = {v: k for k, v in label_map.items()}
