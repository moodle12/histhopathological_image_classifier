import cv2
import numpy as np
import matplotlib.pyplot as plt
from fastapi import File, UploadFile, HTTPException
import tensorflow as tf
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import base64
from fastapi.staticfiles import StaticFiles

app = FastAPI()
# Assume the 'MODEL' variable is already defined with the loaded model
MODEL = tf.keras.models.load_model('model4.h5')
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    content = open("index.html", "r").read()
    return HTMLResponse(content=content, status_code=200, headers={"Content-Type": "text/html"})

def preprocess_input_image(image_path):
    # Load and preprocess the input image for the model
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_mask(image_path):
    input_image = preprocess_input_image(image_path)
    predicted_mask = MODEL.predict(input_image)
    return predicted_mask

def array_to_base64_image(array):
    pil_image = (array * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', pil_image)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded file as bytes
        contents = await file.read()

        # Decode the bytes to an image array using OpenCV
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Preprocess the image for prediction
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        read = img
        img = cv2.resize(img, (512, 512))
        img = img / 255.0  # Normalize pixel values to [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Perform prediction
        predicted_mask = MODEL.predict(img)

        # Convert the predicted mask to base64
        base64_mask = array_to_base64_image(predicted_mask[0, :, :, 0])

        # Convert the uploaded image to base64
        base64_uploaded = array_to_base64_image(read)
        
        return {"prediction": base64_mask, "uploaded": base64_uploaded}
    except Exception as e:
        return {"error": str(e)}


# @app.post("/predict")
# async def predict_image(file: UploadFile = File(...)):
#     try:
#         # Read the uploaded image
#         contents = await file.read()
#         nparr = np.fromstring(contents, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Preprocess the image
#         desired_width, desired_height = 512, 512
#         input_image = cv2.resize(img, (desired_width, desired_height))
#         input_image = input_image / 255.0
#         input_image = np.expand_dims(input_image, axis=0)

#         # Use model.predict to obtain the predicted mask
#         predicted_mask = MODEL.predict(input_image)

#         # Visualize the original image and predicted mask (optional)
#         plt.subplot(1, 2, 1)
#         plt.imshow(img)
#         plt.title('Original Image')
#         plt.axis('off')

#         plt.subplot(1, 2, 2)
#         plt.imshow(predicted_mask[0, :, :, 0], cmap='gray')
#         plt.title('Predicted Mask')
#         plt.axis('off')

#         plt.show()

#         # You can return the predicted mask as a response
#         return {"prediction": predicted_mask.tolist()}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
