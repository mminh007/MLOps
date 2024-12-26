import os
import mlflow.artifacts
import mlflow.artifacts
import yaml
import mlflow
import torch
from mlflow import MlflowClient
from dataclasses import dataclass, field
from controller.model import MultiTaskModel

@dataclass
class Config:
    registered_name: str = "multi-tasks-segmentation"
    model_alias: str = "production"
    

def load_config(tracking=None,
                local_directory=None):
    print("Loading config model and artifacts")
    mlflow.set_tracking_uri(tracking)
    
    try:
        config = Config()
        client = MlflowClient()
        alias_mv = client.get_model_version_by_alias(config.registered_name,
                                                     config.model_alias)
        
        print("Downloading model artifacts with run_id: ", alias_mv.run_id)
        
        model_path_uri = f"runs:/{alias_mv.run_id}/model"
        mlflow.artifacts.download_artifacts(model_path_uri, dst_path=local_directory)
        
        cfg_path_uri = f"runs:/{alias_mv.run_id}/config"
        mlflow.artifacts.download_artifacts(cfg_path_uri, dst_path=local_directory)
        
        with open(local_directory / "multitasks_config.yaml", "rb") as f:
              cfg = yaml.safe_load(f)
              
    except Exception as e:
         print(f"Error loading model artifacts")
         print(e)
         
    return cfg, alias_mv.run_id

        
LOCAL_ARTIFACTS = "DATA/artifacts"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_LOCAL_DIR = "/DATA/mlflow_download"

if not os.path.exists(MLFLOW_LOCAL_DIR):
	MLFLOW_LOCAL_DIR.mkdir(parents = True, exist_ok = True)


def load_model_from_mlflow():
    """
    Get model's weight and config from mlflow 
    ----------------------
    Return:
		Dictionary:
          {
			"config": config.yaml
            "model": model arch + weights
          }
    """

    box = {}
    cfg, _ = load_config(tracking=MLFLOW_TRACKING_URI,
					  local_directory=MLFLOW_LOCAL_DIR)
    
    model = MultiTaskModel(config=cfg)
    
    ckpt = torch.load(MLFLOW_LOCAL_DIR / "best_model.pth")
    model.load_state_dict(ckpt["model_state_dict"])
    
    box["cfg"] = cfg
    box["model"] = model
    return box
    


    

