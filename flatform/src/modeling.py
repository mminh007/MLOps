import torch
import segmentation_models_pytorch as smp
import torchvision
from kornia.losses import focal_loss
import torch.nn as nn
import torch.optim as optim
import os
import mlflow
import numpy as np
from pathlib import Path
from flatform.src.ultis import load_config, connect_mlflow
from flatform.src.data import build_data


def seg_models(config):
    
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


def cls_models(config):

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


class MultiTasksModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        #self.seg_model = seg_models(config)
        self.cls_model = cls_models(config)

    
    def forward(self, x):
        #seg_out = self.seg_model(x)
        cls_out = self.cls_model(x)

        return cls_out
    

def save_checkpoint(model, optimizer, epoch, epoch_loss, checkpoint_path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
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
        return 1, np.inf


def train_model(**kwargs):
    config = load_config()
    
    SRC = Path(config["data_source"])
    DATA_TRAIN =  SRC / config["src_train"]
    
    path_save = SRC / config["output"] / config["version"] 

    if not os.path.exists(path_save):
        path_save.mkdir(parents = True, exist_ok = True)

    connect_mlflow(tracking_uri=config["tracking_uri"],
                    experiment_name=config["experiment_name"])
    

    # data = split_data(data_dir=config["src_train"])
    # train_data = MultiTaskDataset(datasets=data, transforms=True, size=config["imgsz"])

    device = config["device"]

    model = MultiTasksModel(num_classes=config["num_classes"], channels=config["channels"]).to(device)
    #loss = nn.CrossEntropyLoss()

    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"],
                               weight_decay=config["weight_decay"])
    elif config["optimizer"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"],
                                weight_decay=config["weight_decay"])

    train_loader = build_data(data_dir=DATA_TRAIN,
                              imgsz=config["imgsz"],
                              transforms=True, shuffle=True,
                              batch_size=config["batch_size"],
                              num_workers=config["num_workers"])


    with mlflow.start_run(run_name=config["run_name"]) as run:
        print(f"MLFLOW run_id: {run.info.run_id}")
        print(f"MLFLOW experiment_id: {run.info.experiment_id}")
        print(f"MLFLOW run_name: {run.info.run_name}")

        mlflow.set_tag(
            {
                "Model version": config["version"],
                "Architecture": config["arc"],

            }
        )

        mlflow.log_params(
            {
                "input_size": config["imgsz"],
                "arch": config["arch"],
                "encoder_name": config["encoder_name"],
                "cls_classes": config["cls_classes"],
                "lr": config["lr"],
                "batch_size": config["batch_size"],
                "epochs": config["epochs"],
            }
        )

        if config["ckpt_path"] is not None:
            ckpt_path = path_save / config["ckpt_path"]
            start_epoch, best_loss = load_checkpoint(model, optimizer, ckpt_path)
        else:
            start_epoch, best_loss = 1, np.inf
    
        best_model_info = {
            "model_state_dict": None,
            "optimizer_state_dict": None,
            "best_loss": None,
            "epoch": None
        }

        for epoch in range(start_epoch, 1 + config["epochs"]):
            model.train()
            running_loss = 0

            for batch in train_loader:
                label, image, mask, name = batch
                label = label.to(device)
                image = image.to(device)
                #mask = mask.to(device)

                cls_out = model(image)
                #seg_out = seg_out[0].squeeze()
                
                #seg_loss = loss(seg_out, mask)
                cls_loss = focal_loss(cls_out, label, alpha=0.25, gamma=2, reduction="mean")
                #train_loss = config["alpha"] * seg_loss + (1 - config["alpha"] * cls_loss)

                optimizer.zero_grad()
                cls_loss.backward()
                optimizer.step()

                running_loss += cls_loss.item()

            epoch_loss = running_loss / len(train_loader)
            mlflow.log_metric("training_loss", f"{epoch_loss:.6f}", step=epoch)

            val_loss = evaluate_model(model, cls_loss, config["device"])
            mlflow.log_metric("val_loss", f"{val_loss:.6f}", step=epoch)

            #print(f"Epoch {epoch + 1}, train loss: {epoch_loss:.6f}")

            # save_checkpoint(model, optimizer, epoch, epoch_loss, ckpt_path)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_info.update({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": epoch_loss,
                        "epoch": epoch
                    })
                
        exp_path = path_save / run.info.run_id 
        exp_path.mkdir(parents = True, exist_ok = True)
        torch.save(best_model_info, exp_path / "best_model.pth")

        kwargs["ti"].xcom_push(key="run_id", value= run.info.run_id)
        kwargs["ti"].xcom_push(key="val_loss", value=best_model_info["best_loss"])
        print("Training Complete!")        
            
    
    
def evaluate_model(model, criterion, device):
    config = load_config()
    
    SRC = Path(config["data_source"])
    DATA_VAL = SRC / config["src_val"]
    
    if os.path.exists(DATA_VAL):
        config = load_config()

        val_loader = build_data(data_dir=DATA_VAL,
                              imgsz=config["imgsz"],
                              transforms=True, shuffle=True,
                              batch_size=config["batch_size"],
                              num_workers=config["num_workers"])
        
        model.to(device)
        model.eval()
        running_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                label, image, _, _ = batch
                label = label.to(device)
                image = image.to(device)
                #mask = mask.to(device)

                cls_out = model(image)
                # seg_out = seg_out[0].squeeze() 

                # seg_loss = criterion(seg_out, mask)
                cls_loss = focal_loss(cls_out, label, alpha=0.25, gamma=2, reduction="mean")
                #val_loss = config["alpha"] * seg_loss + (1 - config["alpha"] * cls_loss)

                running_loss += cls_loss.item()

        return running_loss / len(val_loader)
    
    else:
        print(f"No validation data path at {DATA_VAL}")