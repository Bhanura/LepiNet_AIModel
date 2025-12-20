import torch
import timm
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
import io

app = FastAPI()

# --- CONFIGURATION ---
MODEL_FILE = "model.pth"              # You must rename your .pth file to this
TRAIN_CSV = "butterfly_images.csv"    # The file you used for training
NAMES_CSV = "sri_lanka_butterflies_245.csv" # The file with English names

print("Loading LepiNet AI...")

# 1. Rebuild the Class Index (CRITICAL)
# We must recreate the exact same LabelEncoder used during training
# to know that Index 0 = "b001", Index 1 = "b002", etc.
try:
    df_train = pd.read_csv(TRAIN_CSV)
    le = LabelEncoder()
    # Fit on the 2nd column (butterfly_id) just like in your notebook
    le.fit(df_train.iloc[:, 1]) 
    CLASSES = list(le.classes_) 
    print(f"✅ Classes loaded: {len(CLASSES)} species found.")
except Exception as e:
    print(f"❌ Error loading training CSV: {e}")
    CLASSES = []

# 2. Load English Names Mapping
try:
    df_names = pd.read_csv(NAMES_CSV)
    # Create dict: {'b001': 'Tailed Jay', ...}
    ID_TO_NAME = dict(zip(df_names['butterfly_id'], df_names['common_name_english']))
    print(f"✅ Names loaded: {len(ID_TO_NAME)} English names found.")
except Exception as e:
    print(f"❌ Error loading names CSV: {e}")
    ID_TO_NAME = {}

# 3. Load the Model
DEVICE = torch.device("cpu") # Hugging Face Free Tier is CPU only
try:
    # Creating model structure (MobileNetV4 Small)
    model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=False, num_classes=len(CLASSES))
    
    # Load your trained weights
    state_dict = torch.load(MODEL_FILE, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 4. Define Transforms
# Must match validation transforms from your notebook
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"status": "running", "models_loaded": len(CLASSES) > 0}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not CLASSES:
        return {"error": "Model classes not loaded properly."}

    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Decode result
        idx = predicted_idx.item()
        species_id = CLASSES[idx] 
        species_name = ID_TO_NAME.get(species_id, species_id) # Fallback to ID if name missing
        conf_score = confidence.item()

        return {
            "species_id": species_id,
            "species_name": species_name,
            "confidence": round(conf_score, 4)
        }
    
    except Exception as e:
        return {"error": str(e)}