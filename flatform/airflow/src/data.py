from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


class LoadDataset():
    def __init__(self, train_dir, test_dir, imgsz, seed, val_size):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.height = imgsz
        self.width = imgsz
        self.seed = seed
        self.val_size = val_size

    def _reshape(self, data):
        return data.reshape(-1, 1, self.height, self.width).astype(np.float32)

    def split_data(self):
        train_df = pd.read_csv(self.train_dir)
        test_df = pd.read_csv(self.test_dir)

        train_df = train_df.drop("ID", axis = 1)
        test_df = test_df.drop("ID", axis = 1)

        x_train, x_val, y_train, y_val = train_test_split(train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values,
                                                          test_size=self.val_size, random_state=self.seed)

        x_train = self._reshape(x_train)
        x_val = self._reshape(x_val)
        x_test = self._reshape(test_df.values)

        return x_train, x_val, x_test, y_train, y_val
    

class DataGenerator(Dataset):
    def __init__(self, x, y, imgsz):
        self.x = x
        self.y = y
        self.transform_image = A.Compose([
                A.Resize(imgsz, imgsz),
                A.HorizontalFlip(p = 0.8),
                A.Blur(),
                A.RandomBrightnessContrast(p = 0.8),

            ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx] # (1,H,W) 
        label = self.y[idx]
        image = image / 255
        image = self.transform_image(image = image)["image"]
        return image, label
    

def build_dataset(args):

    TMP_DIR = Path(args.tmp_dir)
    TRAIN_DIR = TMP_DIR / args.train_file
    VAL_DIR = TMP_DIR / args.val_file

    load_ds = LoadDataset(train_dir=TRAIN_DIR,
                          test_dir=VAL_DIR,
                          imgsz= args.imgsz,
                          seed = args.seed,
                          val_size=args.val_size)
    
    x_train, x_val, x_test, y_train, y_val = load_ds.split_data()

    train_data = DataGenerator(x_train, y_train, args.imgsz)
    val_data = DataGenerator(x_val, y_val, args.imgsz)

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.imgsz, shuffle=True)  

    return train_loader, val_loader, x_test

