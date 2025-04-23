import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms as T
from tqdm import tqdm
from PIL import Image

class CUBDataset(Dataset):

    def __init__(self, csv_file, data_dir, transform=None):

        super().__init__()
        self.csv_file = csv_file
        self.data_dir = data_dir
        self.transform = transform

        # two different methods for loading data; loading in advance or on the fly
        ## Going to load CSV file (labels) at once, but load the images as needed

        self.samples = []
        assert Path(self.csv_file).exists(), f"CSV file {self.csv_file} does not exist"
        with open(self.csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)
            

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx] # this is a dictionary

        image_filename = Path(row["filename"])

        img_path = self.data_dir / "images" / image_filename

        # load the image
        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        class_id = int(row["class_id"]) - 1

        x_min, y_min, x_max, y_max = float(row["x_min"]), float(row["y_min"]), float(row["x_max"]), float(row["y_max"])
        bounding_box = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)

        # want to return the image, the class id, and the bounding box
        return img, (class_id, bounding_box)
        
class CUBDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        if transform is None:
            transform = T.ToTensor()
        self.transform = transform


    def setup(self, stage=None):
        # if stage == 'fit': --> Do something for training
        # if stage == 'test': --> Do something for testing
        self.train_dataset = CUBDataset(
            csv_file=self.data_dir / "train.csv",
            data_dir=self.data_dir,
            transform=self.transform
        )

        self.val_dataset = CUBDataset(
            csv_file=self.data_dir / "val.csv",
            data_dir=self.data_dir,
            transform=self.transform
        )


    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=15,
            persistent_workers=True
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=15,
            persistent_workers=True
        )

    