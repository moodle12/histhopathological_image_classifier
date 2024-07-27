import cv2
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
import tensorflow as tf
from fastapi.staticfiles import StaticFiles
import base64

app = FastAPI()

# Load multiple models
MODELS = {
    'Reinhard': tf.keras.models.load_model('Reinhard.h5'),
    'UNET': tf.keras.models.load_model('model4.h5'),
    'Vahadane': tf.keras.models.load_model('Vahadane.h5'),
    'Macenko': tf.keras.models.load_model('Macenko.h5'),
}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    content = open("index.html", "r").read()
    return HTMLResponse(content=content, status_code=200, headers={"Content-Type": "text/html"})

def preprocess_input_image(image, image_size):
    img = cv2.resize(image, (image_size, image_size))
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_mask(image, model, image_size):
    input_image = preprocess_input_image(image, image_size)
    predicted_mask = model.predict(input_image)
    return predicted_mask

def array_to_base64_image(array):
    pil_image = (array * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', pil_image)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...), 
    model_name: str = Form(...),
    image_size: int = Form(512)
):
    try:
        # Read the uploaded file as bytes
        contents = await file.read()

        # Decode the bytes to an image array using OpenCV
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        read = img.copy()

        # Check if the specified model is available
        if model_name not in MODELS:
            raise HTTPException(status_code=400, detail="Model not found")

        # Get the selected model
        model = MODELS[model_name]

        # Preprocess the image for prediction
        img = preprocess_input_image(img, image_size)

        # Perform prediction
        predicted_mask = model.predict(img)

        # Convert the predicted mask to base64
        base64_mask = array_to_base64_image(predicted_mask[0, :, :, 0])

        # Convert the uploaded image to base64
        base64_uploaded = array_to_base64_image(read)
        
        return {"prediction": base64_mask, "uploaded": base64_uploaded}
    except Exception as e:
        return {"error": str(e)}
