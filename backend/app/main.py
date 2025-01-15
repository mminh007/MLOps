from fastapi import FastAPI
from app.api.v1.routes import pred

app = FastAPI()
app.include_router(router=pred.router, prefix="/v1")