import os
import glob
import sys
import importlib.util
import torch
import torch.nn.functional as F
import pandas as pd
from facenet_pytorch import MTCNN
from PIL import Image


# Define core directories for datasets, ground-truth bona fide images, and experiment outputs
BASE_DIR = "/content/drive/MyDrive/HCML_Project/MAD22_Data/extracted_images/original_sorted"
BONAFIDE_DIR = os.path.join(BASE_DIR, 'BonaFide')
RESULTS_DIR = "./final_experiments_results"
OUTPUT_CSV = "./final_comparison_metrics.csv"

# Set the strict FR verification threshold (tau=0.321) derived from the CASIA-webFace genuine distribution
THRESHOLD = 0.321 

print("Loading ElasticFace Model for Evaluation...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, '/content/drive/MyDrive/HCML_Project/NegFaceDiff/face_recognition_training')

# Init the MTCNN face detector and the pretrained ElasticFace model for biometric evaluation
mtcnn = MTCNN(image_size=112, margin=0, keep_all=False, post_process=False, device=device)
spec = importlib.util.spec_from_file_location("iresnet_module", '/content/drive/MyDrive/HCML_Project/NegFaceDiff/face_recognition_training/backbones/iresnet.py')
iresnet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iresnet)

fr_model = iresnet.iresnet100(num_features=512, use_se=False).to(device)
fr_model.load_state_dict(torch.load('/content/drive/MyDrive/HCML_Project/NegFaceDiff/output/ElasticCos.pth', map_location=device, weights_only=False))
fr_model.eval()



# Extract and L2-normalize high-dimensional biometric embeddings from the input face image
def get_feature(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        cropped = mtcnn(img)
        if cropped is None: return None
        tensor = cropped.unsqueeze(0).float().div_(255).sub_(0.5).div_(0.5).to(device)
        with torch.no_grad(): feat = fr_model(tensor)
        return torch.nn.functional.normalize(feat, p=2, dim=1)
    except:
        return None

results_data = []

if not os.path.exists(RESULTS_DIR):
    print("Results directory not found. Please run experiments first.")
    sys.exit()


# Discover all evaluated architectures from the results directory to begin batch evaluation
models_tested = [m for m in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, m))]


# compute cosine similarities against reference identities
for model_name in models_tested:

    model_path = os.path.join(RESULTS_DIR, model_name)
    datasets = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
    
    for dataset in datasets:

        dataset_path = os.path.join(model_path, dataset)
        morph_folders = [m for m in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, m))]
        
        for morph_name in morph_folders:
            morph_path = os.path.join(BASE_DIR, dataset, f"{morph_name}.jpg")

            if not os.path.exists(morph_path): morph_path = morph_path.replace('.jpg', '.png')
                
            id1 = morph_name.split('-vs-')[0]
            id2 = morph_name.split('-vs-')[1]
            
            try:

                p_minus_path = glob.glob(os.path.join(BONAFIDE_DIR, f'{id1}*'))[0]
                target_path = glob.glob(os.path.join(BONAFIDE_DIR, f'{id2}*'))[0]

            except:
                continue
                
            feat_p_minus, feat_target, feat_morph = get_feature(p_minus_path), get_feature(target_path), get_feature(morph_path)
            
            
            if any(f is None for f in [feat_p_minus, feat_target, feat_morph]): continue
                
            base_sim_target = F.cosine_similarity(feat_morph, feat_target).item()
            base_sim_p_minus = F.cosine_similarity(feat_morph, feat_p_minus).item()
                
            for w in ["w_0", "w_5", "w_10"]:
                gen_img_path = os.path.join(dataset_path, morph_name, w, "generated.png")
                if not os.path.exists(gen_img_path): continue
                    
                feat_gen = get_feature(gen_img_path)
                if feat_gen is None: continue
                    
                sim_target = F.cosine_similarity(feat_gen, feat_target).item()
                sim_p_minus = F.cosine_similarity(feat_gen, feat_p_minus).item()
                
                # Determine if the known identity has been successfully suppressed below the verification threshold
                is_unpaired = "Yes" if sim_p_minus < THRESHOLD else "No"
                
                results_data.append({
                    "Architecture": model_name,
                    "Dataset": dataset,
                    "Morph_ID": morph_name,
                    "Weight": w,
                    "Morph_Sim_Target": round(base_sim_target, 4),
                    "Morph_Sim_P-": round(base_sim_p_minus, 4),
                    "Gen_Sim_Target": round(sim_target, 4),
                    "Gen_Sim_P-": round(sim_p_minus, 4),
                    "Unpaired": is_unpaired
                })


# Aggregate all computed metrics into a structured dataframe and export to CSV ... 
if results_data:
    df = pd.DataFrame(results_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n Full evaluation complete! {OUTPUT_CSV} created.")
