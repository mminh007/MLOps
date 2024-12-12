import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
import os
import numpy as np
import albumentations as A
import cv2
from airflow import DAG
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import segmentation_models_pytorch as smp
from kornia.losses import focal_loss
from airflow.operators.python import PythonOperator


DATA_SOURE = Path("./DATA")
DATA_TRAIN = DATA_SOURE / "train" 
DATA_VAL = DATA_SOURE / "val"
ARTIFACTS = DATA_SOURE / "artifacts"

default_args = {

}


def load_config():
    config_path = "./config/multitask_config.yaml"
    if not os.path.exists(config_path):
        raise FileExistsError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config
                              

def create_temp_dir():
    config = load_config()
    cuda_available = True if torch.cuda.is_available() else False
    print(f"Using GPU: {cuda_available}")

    path_save = Path("./output_dir") / config["version"]
    path_save.mkdir(parents = True, exist_ok = True)

    if not path_save.exists():
        raise FileNotFoundError(f"Failed to create directory: {path_save}")


def split_data(data_dir):
    config = load_config()

    data_dir = data_dir / config["version"]

    benign, malignant, normal = [], [], []

    for mask in (data_dir / "benign").iterdir():
        if "_mask" in mask.name:
            image = data_dir / "benign" / mask.name.replace("_mask.png", ".png")
            malignant.append((0, mask, image))

    for mask in (data_dir / "malignant").iterdir():
        if "_mask" in mask.name:
            image = data_dir / "malignant" / mask.name.replace("_mask.png", ".png")
            benign.append((0, mask, image))

    for mask in (data_dir / "normal").iterdir():
        if "_mask" in mask.name:
            image = data_dir / "normal" / mask.name.replace("_mask.png", ".png")
            normal.append((0, mask, image))

    all_data = benign + malignant + normal
    labels = [item[0] for item in all_data]

    if config["USE_KFOLDS"] == 1:
        kf = StratifiedKFold(n_splits = config["n_splits"]) #N_SPLITS

        folds = []

        # Splitting data into folds
        for train_index, val_index in kf.split(np.zeros(len(labels)), labels):
            train_set = [all_data[i] for i in train_index]
            val_set = [all_data[i] for i in val_index]
            folds.append((train_set, val_set))

        return folds
    
    else:
        return all_data
        

class MultiTaskDataset(Dataset):
    def __init__(self, datasets, size = 448, transforms=None):
        super().__init__()
        self.datasets = datasets
        self.size = size
    
        self.transforms = transforms
        
        self.transform_image = A.Compose(
            [
                A.Resize(height = self.size, width = self.size, interpolation = cv2.INTER_LINEAR),
                A.HorizontalFlip(p = 0.6),
                A.Blur(),
                A.RandomBrightnessContrast(p = 0.6),
                A.CoarseDropout(p = 0.6, max_holes=18, max_height=24, max_width=24, min_holes=12, min_height=12, min_width=12, fill_value=0),
                A.Normalize(),
            ]
        )


    def __len__(self):
        return len(self.datasets)
    

    def __getitem__(self, index):
        label, mask_path, image_path = self.datasets[index]
        
        name = image_path.name.split(".")[0]
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            transformed = self.transform_image(image=image, mask=mask)
            transformed_image = transformed["image"]
            transfromed_mask = transformed["mask"]
        

        tensor_image = torch.from_numpy(transformed_image).permute(2,0,1).to(torch.float32)

        return label, tensor_image, transfromed_mask, name
    

def seg_models():
    config = load_config()
    
    aux_param = {
        "pooling": "avg",
        "dropout": 0.5,
        "classes": config["cls_classes"]
    }

    params = {
        "encoder_name": config["encoder_name"],
        "encoder_depth": config["encoder_depth"],
        "encoder_weights": config["encoder_weights"],
        "in_channels": config["in_channels"],
        "classes": config["seg_classes"],
        "aux_params": aux_param,
    }

    if config["arch"] == "unet":
        return smp.Unet(**params)
    
    if config["arch"] == "unetpp":
        return smp.UnetPlusPlus(**params)
    
    if config["arch"] == "deeplabv3plus":
        return smp.DeepLabV3Plus(**params)
    
    if config["arch"] == "fpn":
        return smp.FPN(**params)


def cls_models():
    config = load_config()

    if config["encoder_name"] == "restnet50":
        model = torchvision.models.resnet50(weights = "DEFAULT")

    elif config["encoder_name"] == "resnext50_32x4d":
        model = torchvision.models.resnext50_32x4d(weights = "DEFAULT")

    elif config["encoder_name"] == "tu-wide_resnet50_2":
        model = torchvision.models.wide_resnet50_2(weights = "DEFAULT")
    else:
        state_dict = torch.hub.load_state_dict_from_url(config["efficientnet-b4"])
        model = torchvision.models.efficientnet_b4(weights = "DEFAULT")
        model.load_state_dict(state_dict)
        num_features = model.classifier[1].in_features
        model.classifier = torch.nn.Linear(num_features, out_features=3) 


    if config["encoder_name"] != "efficientnet-b4":
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, out_features=3)

        for param in model.layer4.parameters():
            param.requires_grad = True
    
    return model


class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_model = seg_models()
        self.cls_model = cls_models()

    
    def forward(self, x):
        seg_out = self.seg_model(x)
        cls_out = self.cls_model(x)

        return seg_out, cls_out
    

def save_checkpoint(model, optimizer, epoch, epoch_loss, checkpoint_path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimzier_state_dict": optimizer.state_dict(),
        "best_loss": epoch_loss,
        "epoch": epoch
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]

        print(f"Checkpoint loaded from {checkpoint_path}, starting at epoch {start_epoch}")
        return start_epoch, best_loss
    
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
        return 1, None


def train_model():
    config = load_config()

    path_save = Path("./output_dir") / config["version"]

    ckpt_path = path_save / config["ckpt_path"]
    model_path = path_save / "best_model.pth"

    data = split_data(data_dir=DATA_TRAIN)

    train_data = MultiTaskDataset(datasets=data, transforms=True, size=config["imgsz"])

    device = config["device"]

    model = MultiTaskModel().to(device)
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)

    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"],
                               weight_decay=config["weight_decay"])
    elif config["optimizer"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"],
                                weight_decay=config["weight_decay"])
        
    train_loader = DataLoader(train_data, shuffle=True, batch_size=config["batch_size"],
                              num_workers=config["num_workers"])


    if config["ckpt_path"] is not None:
        start_epoch, best_loss = load_checkpoint(model, optimizer, ckpt_path)
    else:
        start_epoch, best_loss = 1, None


    for epoch in range(start_epoch, 1 + config["epochs"]):
        model.train()
        running_loss = 0

        for batch in train_loader:
            label, image, mask, name = batch
            label = label.to(device)
            image = image.to(device)
            mask = mask.to(device)

            seg_out, cls_out = model(image)
            seg_out = seg_out[0].squeeze()
            
            seg_loss = dice_loss(seg_out, mask)
            cls_loss = focal_loss(cls_out, label, alpha=0.25, gamma=2, reduction="mean")
            train_loss = config["alpha"] * seg_loss + (1 - config["alpha"] * cls_loss)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, train loss: {epoch_loss: .6f}")

        save_checkpoint(model, optimizer, epoch, epoch_loss, ckpt_path)
        
        if best_loss == None:
            best_loss = epoch_loss
            save_checkpoint(model, optimizer, epoch, epoch_loss, model_path)
        
        else:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_checkpoint(model, optimizer, epoch, epoch_loss, model_path)

        
def validate_model():

    if os.path.exists(DATA_VAL):
        config = load_config()

        path_save = Path("./output_dir") / config["version"]

        data = split_data(data_dir=DATA_VAL)

        val_data = MultiTaskDataset(data, transforms=True)
        val_loader = DataLoader(val_data, batch_size=config["batch_size"],
                                shuffle=True, num_workers=config["num_workers"])

        device = config["device"]
        model = MultiTaskModel()

        ckpt = torch.load(path_save / "best_model.pth", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"]).to(device)
        dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)

        model.eval()
        running_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                label, image, mask, name = batch
                label = label.to(device)
                image = image.to(device)
                mask = mask.to(device)

                seg_out, cls_out = model(image)
                seg_out = seg_out[0].squeeze() 

                seg_loss = dice_loss(seg_out, mask)
                cls_loss = focal_loss(cls_out, label, alpha=0.25, gamma=2, reduction="mean")
                val_loss = config["alpha"] * seg_loss + (1 - config["alpha"] * cls_loss)

                running_loss += val_loss.item()

        return running_loss / len(val_loader)
    
    else:
        print(f"No validation data path at {DATA_VAL}")


def logging_artifacts():
    config = load_config()
    
    path_save = Path("./output_dir") / config["version"]
    artifacts = ARTIFACTS / "multi-tasks" / config["version"]
    artifacts.mkdirs(parents = True, exist_ok = True)

    # copy config file
    cmd_0 = f"cp ./config/multitasks_config.yaml {artifacts}/multitask_config.yaml"

    # copy output path
    cmd_1 = f"cp -r {path_save} {artifacts}"

    for cmd in [cmd_0, cmd_1]:
        os.system(cmd)


with DAG(
    "Multi-tasks_Predictions",
    default_args= default_args,
    description= "A simple ML pipeline",
) as dag:

    create_temp_dir_task = PythonOperator(
        task_id='create_temp_dir',
        python_callable=create_temp_dir,
        dag=dag,
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        dag=dag,
    )

    validate_model_task = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model,
        dag=dag,
    )

    logging_artifacts_task = PythonOperator(
        task_id='logging_artifacts',
        python_callable=logging_artifacts,
        dag=dag,
    )

    create_temp_dir_task  >> train_model_task >> validate_model_task >> logging_artifacts_task