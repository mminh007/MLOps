from dataclasses import dataclass, field

@dataclass
class CLS_Config:
	registered_name:str =  "restnet50"
	model_alias: str =  "productions"
	model_version: str = "v1"