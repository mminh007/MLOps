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

