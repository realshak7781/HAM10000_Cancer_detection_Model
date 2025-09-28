import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO

# --- 1. Load the trained model ---
# We load the model once when the application starts.
MODEL = tf.keras.models.load_model("skin_cancer_model.keras")
CLASS_NAMES = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions ', 'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']
IMG_WIDTH = 100
IMG_HEIGHT = 75

# --- 2. Preprocess the image ---
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocesses the uploaded image to the format the model expects.
    """
    # Open the image file from bytes
    image = Image.open(BytesIO(image_bytes))
    
    # Resize the image
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Convert to NumPy array
    image_array = np.asarray(image)
    
    # Normalize the image
    image_array = image_array / 255.0
    
    # Add a batch dimension
    return np.expand_dims(image_array, axis=0)

# --- 3. Make a prediction ---
def predict(image_bytes: bytes) -> dict:
    """
    Takes image bytes, preprocesses the image, and returns the prediction.
    """
    # Preprocess the image
    processed_image = preprocess_image(image_bytes)
    
    # Get prediction probabilities
    predictions = MODEL.predict(processed_image)
    
    # Get the predicted class index and confidence
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # Get the class name
    predicted_class_name = CLASS_NAMES[predicted_index]
    
    return {
        "predicted_class": predicted_class_name,
        "confidence": float(confidence)
    }