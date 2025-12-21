import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from supabase import create_client, Client
from huggingface_hub import HfApi
import timm
import shutil
import requests
import json

# CONFIG
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") # Service Role Key
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "bhanura/lepinet-backend" # <--- UPDATE THIS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fine_tune_model():
    print("🚀 Starting Automated Fine-Tuning...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 1. Fetch ALL Species IDs (To determine Class List)
    print("Fetching species list...")
    species_res = supabase.table("species").select("butterfly_id, common_name_english").execute()
    # Create a sorted list of ALL 245 IDs: ['b001', 'b002', ..., 'b245']
    all_class_ids = sorted([row['butterfly_id'] for row in species_res.data])
    
    # Map Common Names to IDs (e.g., "Common Rose" -> "b001")
    name_to_id = {row['common_name_english']: row['butterfly_id'] for row in species_res.data}
    
    # Map ID to Index (for the AI)
    id_to_idx = {cls_id: idx for idx, cls_id in enumerate(all_class_ids)}
    num_classes = len(all_class_ids) # Should be 245

    print(f"✅ Universe defined: {num_classes} total species.")

    # 2. Fetch 'READY' Training Data
    reviews = supabase.table("expert_reviews")\
        .select("id, agreed_with_ai, identified_species_name, ai_logs(image_url, predicted_species_name)")\
        .eq("training_status", "ready")\
        .execute().data
    
    if len(reviews) == 0:
        return "No images marked 'ready' for training."

    # 3. Download & Organize
    data_dir = "training_data"
    if os.path.exists(data_dir): shutil.rmtree(data_dir)
    os.makedirs(data_dir)
    
    processed_ids = []
    
    print(f"Downloading {len(reviews)} images...")
    for row in reviews:
        # Determine Name
        name = row['ai_logs']['predicted_species_name'] if row['agreed_with_ai'] else row['identified_species_name']
        
        # Convert Name -> ID (b001)
        if name in name_to_id:
            cls_id = name_to_id[name]
            save_dir = os.path.join(data_dir, cls_id)
            os.makedirs(save_dir, exist_ok=True)
            
            # Download
            try:
                img_resp = requests.get(row['ai_logs']['image_url'], timeout=5)
                if img_resp.status_code == 200:
                    with open(os.path.join(save_dir, f"{row['id']}.jpg"), 'wb') as f:
                        f.write(img_resp.content)
                    processed_ids.append(row['id'])
            except:
                pass

    # 4. Initialize Model (Handling Size Change)
    print(f"Building Model for {num_classes} classes...")
    model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=True, num_classes=num_classes)
    
    # Try to load old weights (Partial Load)
    if os.path.exists("model.pth"):
        print("Loading existing weights (Partial)...")
        old_state = torch.load("model.pth", map_location=DEVICE)
        
        # Filter out the "Head" (Classifier) because size might have changed (7 -> 245)
        new_state = model.state_dict()
        pretrained_dict = {k: v for k, v in old_state.items() if k in new_state and v.size() == new_state[k].size()}
        
        new_state.update(pretrained_dict)
        model.load_state_dict(new_state)
    
    model.to(DEVICE)
    model.train()

    # 5. Train
    transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Custom Dataset to ensure we use our specific class mapping
    train_dataset = datasets.ImageFolder(data_dir, transform=transform)
    # Force the dataset to use our sorted ID list as classes
    # (Note: This assumes folders are named 'b001', 'b002' etc.)

    loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    print("Training Loop Starting...")
    for epoch in range(5): # 5 Epochs
        total_loss = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss}")

    # 6. Save & Upload
    torch.save(model.state_dict(), "model.pth")
    
    # Save the Class Map so app.py knows what's what
    with open("classes.json", "w") as f:
        json.dump(all_class_ids, f)

    print("Uploading to Hugging Face...")
    api = HfApi(token=HF_TOKEN)
    api.upload_file(path_or_fileobj="model.pth", path_in_repo="model.pth", repo_id=REPO_ID, repo_type="space")
    api.upload_file(path_or_fileobj="classes.json", path_in_repo="classes.json", repo_id=REPO_ID, repo_type="space")

    # 7. Update Database (Mark as Trained)
    if processed_ids:
        supabase.table("expert_reviews").update({"training_status": "trained"}).in_("id", processed_ids).execute()

    return "Training Complete! Space restarting..."