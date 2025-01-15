from pydantic import BaseModel
from fastapi import File, UploadFile

class InputCfg(BaseModel):
	model_version: str = "v1"
	image: UploadFile = File(...)
	