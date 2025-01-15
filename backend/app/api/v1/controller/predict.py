import torch
import torch.nn as nn
import torchvision
import segmentation_models_pytorch as smp



def seg_models(arch: str = "unet",
               encoder_name: str = "resnet50",
               encoder_depth: int = 5,
               encoder_weights: str= "imagenet",
               in_channels: int = 3,
               n_seg: int = 1,
               n_cls: int = 3):
    
    aux_param = {
        "pooling": "avg",
        "dropout": 0.5,
        "classes": n_cls
    }

    params = {
        "encoder_name": encoder_name,
        "encoder_depth": encoder_depth,
        "encoder_weights": encoder_weights,
        "in_channels": in_channels,
        "classes": n_seg,
        "aux_params": aux_param,
    }

    if arch == "unet":
        return smp.Unet(**params)
    
    if arch == "unetpp":
        return smp.UnetPlusPlus(**params)
    
    if arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(**params)
    
    if arch == "fpn":
        return smp.FPN(**params)


def cls_models(encoder_name: str = "resnet50",
               url: str = None):

    if encoder_name == "resnet50":
        model = torchvision.models.resnet50(weights = "DEFAULT")

    elif encoder_name  == "resnext50_32x4d":
        model = torchvision.models.resnext50_32x4d(weights = "DEFAULT")

    elif encoder_name == "tu-wide_resnet50_2":
        model = torchvision.models.wide_resnet50_2(weights = "DEFAULT")
    else:
        state_dict = torch.hub.load_state_dict_from_url(url)
        model = torchvision.models.efficientnet_b4(weights = "DEFAULT")
        model.load_state_dict(state_dict)
        num_features = model.classifier[1].in_features
        model.classifier = torch.nn.Linear(num_features, out_features=3) 


    if encoder_name != "efficientnet-b4":
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, out_features=3)

        for param in model.layer4.parameters():
            param.requires_grad = True
    
    return model


class MultiTaskModel(nn.Module):
    def __init__(self, encoder_name, encoder_depth, encoder_weights, in_channels, n_seg, n_cls, url=None):
        super().__init__()
        self.seg_model = seg_models(encoder_name, encoder_depth, encoder_weights, in_channels, n_seg, n_cls)
        self.cls_model = cls_models(encoder_name, url)

    
    def forward(self, x):
        seg_out = self.seg_model(x)
        cls_out = self.cls_model(x)

        return seg_out, cls_out


def multi_predict(model_version):

    