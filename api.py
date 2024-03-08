from io import BytesIO

import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile

from src.models.predict_model import predict as model_predict

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())
    label = model_predict(image)

    return {"label": label}

