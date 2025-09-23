from fastapi import FastAPI, UploadFile, File
from predict import predict

app = FastAPI(title="Skin Cancer Detection API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Skin Cancer Detection API!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Receives an image file, makes a prediction, and returns the result.
    """
    # Read the image file as bytes
    image_bytes = await file.read()
    
    # Get the prediction
    prediction_result = predict(image_bytes)
    
    return prediction_result

