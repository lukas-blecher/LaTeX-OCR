from http import HTTPStatus
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
from pix2tex.cli import initialize, call_model

model = None
app = FastAPI(title='pix2tex API')


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


@app.on_event('startup')
async def load_model():
    global model
    if model is None:
        model = initialize()


@app.get('/')
def root():
    '''Health check.'''
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {},
    }
    return response


@app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    global model
    image = Image.open(file.file)
    pred = call_model(*model, img=image)
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': pred,
    }
    return response
