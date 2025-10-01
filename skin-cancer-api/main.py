from fastapi import FastAPI, UploadFile, File
from predict import predict
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="Skin Cancer Detection API")

# Define the list of allowed origins:(jaha se request aaaenge)
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:5500", # Common for local development
    "http://127.0.0.1:5500", # Common for local development
    "null",                  # Sometimes needed for local file:// access
    # Add the URL of your deployed frontend app here once you have it
    # e.g., "https://your-awesome-app.netlify.app", 
]

# Add the CORS middleware to the FastAPI application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

