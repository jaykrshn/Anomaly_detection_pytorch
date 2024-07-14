import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image


def kl_loss(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

class CustomDataset(Dataset):
    def __init__(self, input_dir, label_dir, transform=None):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.transform = transform

        self.input_images = sorted(os.listdir(self.input_dir))
        self.label_images = sorted(os.listdir(self.label_dir))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        label_image_path = os.path.join(self.label_dir, self.label_images[idx])

        input_image = Image.open(input_image_path).convert("RGB")
        label_image = Image.open(label_image_path).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            label_image = self.transform(label_image)

        return input_image, label_image