from fastapi import FastAPI
from app.api.v1 import routes 

app = FastAPI()
app.include_router(router=routes, prefix="/v1")