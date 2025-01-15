from dataclasses import dataclass, field

@dataclass
class MultiTasks_Config:
    version: str = "v0.1"
    model_path: str = field(init=False)
    config_path: str = field(init=False)
    arch: str = "unet"
    encoder_name: str = "resnet50"
    encoder_depth: int = 5
    encoder_weigths: str = "imagenet"
    in_channels: int = 3
    n_seg: int = 1
    n_cls: int = 3
    url: str = field(init=False)

    def __post_init__(self):
        self.model_path = f"/DATA/artifacts/multi-tasks/{self.version}/best_model.pth"
        self.config_path = f"/DATA/artifacts/multi-tasks/{self.version}/multitasks_config.yaml"
        self.url = "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth" \
                     if self.encoder_name == "efficientnet-b4" else None
    
