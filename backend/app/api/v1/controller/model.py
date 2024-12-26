import torch
import segmentation_models_pytorch as smp
from kornia.losses import focal_loss
import torch.nn as nn
import torchvision
from utils import load_config
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
        model.fc = torch.nn.Linear(num_features, out_features=config["cls_classes"])

        for param in model.layer4.parameters():
            param.requires_grad = True
    
    return model


class MultiTaskModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seg_model = seg_models(config)
        self.cls_model = cls_models(config)

    
    def forward(self, x):
        #seg_out = self.seg_model(x)
        cls_out = self.cls_model(x)

        return cls_out #seg_out


# class UnNormalize():
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std
        
        
#     def __call__(self, tensor):
#         for t, m, s in zip(tensor, self.mean, self.std):
#             t.mul_(s).add_(m)
            
#         return tensor

def unormalize(img, mean, std):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
        
    return img

def transform_input(image, size):
    transfrom = A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), 
                            max_pixel_value=255.0),
            ToTensorV2(),
		]
	)
    transform_image = transfrom(image=image)["image"]
    return transform_image


async def predict(artifacts, img):
    """
    artifacts:{
				"cfg": config.yaml
                "model": model arch + weights
            }
    """
    model = artifacts["model"].eval()
    cfg = artifacts["cfg"]
        
    img = transform_input(img, cfg["imgsz"])

    labels = ["Benign", "Maligant", "Normal"]
    with torch.no_grad():
        x = img.unsqueeze(0)
        output = model(x)
             
        predict = output.argmax(1).detach().cpu().numpy()

        predicted_label = labels[predict]

    return {"label": predicted_label,
            "Confidence": torch.max(output)}
    
    
            

    
        
        

        
        