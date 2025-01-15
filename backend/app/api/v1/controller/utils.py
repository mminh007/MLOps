import mlflow
from mlflow import MlflowClient
from configs.cfg import CLS_Config
from dotenv import load_dotenv
import os
import yaml
import mlflow.artifacts
from model import load_model

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")



def load_config():
	print(f"Loading model and artifacts")
	mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
	artifacts = {}

	try:
		deploy_config = CLS_Config()
		client = MlflowClient()
		alias_mv = client.get_model_version_by_alias(deploy_config.registered_name,
											   		 deploy_config.model_alias)

		print(f"Downloading model artifacts with run_id:", alias_mv.run_id)

		config_artifacts_uri = f"run://{alias_mv.run_id}/config"
		mlflow.artifacts.download_artifacts(artifact_uri=config_artifacts_uri,
									  		dst_path=".")
		
		with open("./config/parameters.yaml", "rb") as f:
			config = yaml.safe_load(f)
		
		artifacts["config"] = config
		artifacts["deploy_config"] = deploy_config

		#model_uri = f"models:/{deploy_config.registered_name}@production"

		artifacts["model"] = load_model(config=config,
								  		run_id=alias_mv.run_id)
	
	except Exception as e:
		print(f"Error loading model artifacts")
		print(e)

	return artifacts