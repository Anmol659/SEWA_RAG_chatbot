# tools/pest_detection_tool.py

import torch
from torchvision import transforms, models
from PIL import Image
from collections import OrderedDict
import os

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/best_model.pth"
CLASS_NAMES = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
    'Blueberry__healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy',
    'Corn_(maize)Cercospora_leaf_spot_Gray_leaf_spot', 'Corn(maize)Common_rust', 'Corn_(maize)_Northern_Leaf_Blight',
    'Corn_(maize)healthy', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
    'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach__healthy',
    'Pepper_bell__Bacterial_spot', 'Pepper_bell_healthy', 'Potato_Early_blight', 'Potato__Late_blight',
    'Potato__healthy', 'Raspberry_healthy', 'Rice_Bacterial_leaf_blight', 'RiceBrown_spot', 'Rice_Hispa',
    'Rice_Leaf_blast', 'RiceLeaf_scald', 'RiceNarrow_brown_leaf_spot', 'RiceNeck_blast', 'Rice_Sheath_blight',
    'Rice_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry__healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two-spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato__healthy',
    'Wheat_brown_rust', 'Wheat_healthy', 'Wheat_septoria'
]

# --- Image Transformation ---
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Load Model (Singleton Pattern) ---
def load_model():
    """Loads the pre-trained EfficientNet model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please ensure it is present in the 'models' directory.")
    
    model = models.efficientnet_v2_m(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

MODEL = None
try:
    MODEL = load_model()
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")

# --- Inference Function (The Tool) ---
def get_pest_prediction(image_path: str) -> str:
    """
    Identifies crop diseases from an image of a plant leaf.
    This tool is used when the user provides an image. It helps in the early
    detection of diseases, allowing for timely intervention. The Gemini model
    will automatically invoke this function when it detects an image in the user's prompt.

    Args:
        image_path: The file path to the uploaded image of the crop.

    Returns:
        A string describing the most likely disease and the confidence of the prediction.
    """
    if MODEL is None:
        return "Error: Pest prediction model could not be loaded. Please check the model file path and class list configuration."
        
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = MODEL(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class_idx = torch.max(probabilities, 1)
            
        class_name = CLASS_NAMES[predicted_class_idx.item()]
        confidence_score = confidence.item() * 100
        
        formatted_class = class_name.replace("_", " ").replace("__", " - ")
        
        return (f"Detected Disease: **{formatted_class}** with "
                f"**{confidence_score:.2f}%** confidence.")
                
    except FileNotFoundError:
        return f"Error: Image file not found at path: {image_path}"
    except Exception as e:
        return f"An error occurred during pest prediction: {e}"

if __name__ == "__main__":
    print("--- Testing PestPredict Tool ---")
    
    # Create a dummy image for testing if it doesn't exist
    dummy_image_path = "dummy_leaf.png"
    if not os.path.exists(dummy_image_path):
        try:
            Image.new('RGB', (100, 100), color='green').save(dummy_image_path)
            print(f"Created a dummy image for testing: {dummy_image_path}")
            prediction = get_pest_prediction(dummy_image_path)
            print(prediction)
        except Exception as e:
            print(f"Could not create or test with dummy image: {e}")
    else:
        print(f"Using existing dummy image: {dummy_image_path}")
        prediction = get_pest_prediction(dummy_image_path)
        print(prediction)