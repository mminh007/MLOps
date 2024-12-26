import os
import mlflow
import yaml
import torch
from controller.utils import load_config, load_model_from_mlflow
from controller.model import MultiTaskModel, predict
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from dotenv import load_dotenv


model_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
	model_artifacts["v1"] = load_model_from_mlflow()

	yield
	model_artifacts.clear()


router = APIRouter(lifespan=lifespan)

@router.post("/predict",
			 description="Predict the segmentation")

async def predict_img(input_data, model_version):
	"""
	input_data: image
	model_version: version using predict
	--------------------------------------
	Return:
		prediction
	"""
	try:

		artifacts = model_artifacts.get(model_version)

		if artifacts is None:
			return HTTPException(
				status_code=status.HTTP_400_BAD_REQUEST,
				detail="Invalid model version"
			)
		
		output = await predict(artifacts, input_data)
		if isinstance(output, Exception):
			raise HTTPException(
				status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
				detail=str(output)
			)
		
		return output
	
	except Exception as e:
		return JSONResponse(content={"error": str(e)}, status_code=500)
	



