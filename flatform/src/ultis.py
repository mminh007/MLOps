import torch
import os
import yaml
import mlflow
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from mlflow.tracking import MlflowClient


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

    path_save = Path(config["data_source"]) / config["output"] / config["version"]
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
            benign.append((1, mask, image))

    for mask in (data_dir / "normal").iterdir():
        if "_mask" in mask.name:
            image = data_dir / "normal" / mask.name.replace("_mask.png", ".png")
            normal.append((2, mask, image))

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
    

def logging_artifacts(**kwargs):
    config = load_config()
    
    SRC = Path(config["data_source"])
    ARTIFACTS = SRC / config["artifacts"]
    
    path_save = SRC / config["output"] / config["version"] 
    local_artifacts = ARTIFACTS / config["task"] / config["version"] 
    local_artifacts.mkdir(parents = True, exist_ok = True)

    run_id = kwargs["ti"].xcom_pull(task_ids='train_model', key='run_id')
    val_loss = kwargs["ti"].xcom_pull(task_ids="train_model", key="val_loss")

    client = MlflowClient()
    model_alias = config["model_alias"]
    registered_name = config["registered_name"]

    try:
        model = client.get_model_version_by_alias(name=registered_name,
                                              alias=model_alias)
        print(f"Alias: {model_alias} found")
    
    except:
        print(f"Alias: {model_alias} not found")
        registered_model(client, registered_name, model_alias, run_id)
    
    else:
        print(f"Retrieving run: {model.run_id}")
        prod_metric = mlflow.get_run(model.run_id).data.metrics
        prod_val_loss = prod_metric["val_loss"]

        if prod_val_loss < val_loss:
            print(f"Current model is better: {prod_val_loss}")
        else:
            registered_model(client, registered_name, model_alias, run_id)
    
    # ------ save to local --------
    # copy config file
    cmd_0 = f"cp ./config/multitasks_config.yaml {local_artifacts / run_id}/multitasks_config.yaml"

    # copy output path
    cmd_1 = f"cp -r {path_save / run_id} {local_artifacts}"

    for cmd in [cmd_0, cmd_1]:
        os.system(cmd)
            
    # save to mlflow
    mlflow.log_artifact(path_save / run_id / "best_model.pth", artifact_path="model")
    mlflow.log_artifact("./config/multitasks_config.yaml", artifact_path="config")

def registered_model(client,
                   registered_name: str,
                   model_alias: str,
                   run_id: str):
    try:
        client.create_registered_model(name=registered_name)
        client.get_registered_model(name=model_alias)
    
    except:
        print(f"Model: {registered_name} already exists")
    
    print(f"Create model version: {model_alias}")
    model_uri = f"runs:/{run_id}/pytorch-model"
    mv = client.create_model_version(registered_name, model_uri, run_id)

    # Override 'alias' to the best model version
    print(f"Creating model alias: {model_alias}")
    client.set_registered_model_alias(name=registered_name,
                                        alias=model_alias,
                                        version=mv.version)
    
    print("--Model Version--")
    print("Name: {}".format(mv.name))
    print("Version: {}".format(mv.version))
    print("Aliases: {}".format(mv.aliases))

    

def connect_mlflow(tracking_uri,
                   experiment_name:str):
    MLFLOW_TRACKING_URI = tracking_uri
    MLFLOW_EXPERIMENT_NAME = experiment_name
    
    try:
        mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        print(f"MLFLOW TRACKING URI: {MLFLOW_TRACKING_URI}")
        print(f"MLFLOW EXPERIMENT NAME: {MLFLOW_EXPERIMENT_NAME}")
        
    except Exception as e:
        print(f"Error: {e}")
        
        raise e
        
    