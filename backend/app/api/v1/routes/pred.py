<<<<<<< HEAD
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, HTTPException, status
from app.api.core.config import MultiTasks_Config
from app.api.v1.controller.predict import MultiTasksModel, multi_predict


def load_model(config: MultiTasks_Config):
    model = MultiTasksModel(
        encoder_name=config.encoder_name,
        encoder_depth=config.encoder_depth,
        encoder_weights=config.encoder_weigths,
        in_channels=config.in_channels,
        n_seg=config.n_seg,
        n_cls=config.n_cls,
        url=config.url,
    )

    model_state_dict = torch.load(config.model_path)
    model.load_state_dict(model_state_dict)

    return model


def load_config(version: str):
    print(f"Loading model artifacts for version {version}")

    try:
=======
from PIL import Image
import io
from schema.data_cfg import InputCfg
from controller.utils import load_config
from controller.model import predict
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()


model_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
	"""
	"""
	model_artifacts["v1"] = load_config()
	# update selectbox streamlit if adding model version
	yield
	model_artifacts.clear()


router = APIRouter(lifespan=lifespan)

@router.post("/predict",
			 description="Predict the segmentation")

async def predict_img(data_input: InputCfg):

	"""
	input_data: image
	model_version: version using predict
	--------------------------------------
	Return:
		prediction
	"""
	model_version = data_input.model_version
	image_data = await data_input.image.read()
	image = Image.open(io.BytesIO(image_data))
	try:

		artifacts = model_artifacts.get(model_version)

		if artifacts is None:
			return HTTPException(
				status_code=status.HTTP_400_BAD_REQUEST,
				detail="Invalid model version"
			)
		
		output = await predict(artifacts, image)
		if isinstance(output, Exception):
			raise HTTPException(
				status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
				detail=str(output)
			)
		
		return output
	
	except Exception as e:
		return JSONResponse(content={"error": str(e)}, status_code=500)
	


>>>>>>> 6a2ca5afbef68213c9c8799c848bbee2365b2615

