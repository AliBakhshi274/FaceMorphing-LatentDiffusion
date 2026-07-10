import os
import glob
import sys
import importlib.util
import torch
from facenet_pytorch import MTCNN
from PIL import Image

# Config & Paths 
# ...............................................................................
DATASETS = ["OpenCV", "FaceMorpher", "Webmorph", "MIPGAN_I", "MIPGAN_II"]
BASE_DIR = "/content/drive/MyDrive/HCML_Project/MAD22_Data/extracted_images/original_sorted"
BONAFIDE_DIR = os.path.join(BASE_DIR, 'BonaFide')
CONTEXT_SAVE_PATH = "/content/drive/MyDrive/HCML_Project/NegFaceDiff/output/contexts/random_synthetic_uniform_1.npy"

# Checkpoint paths for each architecture
MODELS = {
    "NegFaceDiff": "/content/drive/MyDrive/HCML_Project/NegFaceDiff/neg_prompt/trained_models/CASIA-IDiff-cpd25",
    "AdaptDiff": "/content/drive/MyDrive/HCML_Project/NegFaceDiff_AdaptDiff/DM_CASIA_cpd25"
}

BASE_OUTPUT = "./final_experiments_results"
os.makedirs(BASE_OUTPUT, exist_ok=True)




# Setup Face Recognition (ElasticFace)
# ...............................................................................
print("Loading ElasticFace Model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.insert(0, '/content/drive/MyDrive/HCML_Project/NegFaceDiff/face_recognition_training')

mtcnn = MTCNN(image_size=112, margin=0, keep_all=False, post_process=False, device=device)
spec = importlib.util.spec_from_file_location("iresnet_module", '/content/drive/MyDrive/HCML_Project/NegFaceDiff/face_recognition_training/backbones/iresnet.py')
iresnet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iresnet)

fr_model = iresnet.iresnet100(num_features=512, use_se=False).to(device)
fr_model.load_state_dict(torch.load('/content/drive/MyDrive/HCML_Project/NegFaceDiff/output/ElasticCos.pth', map_location=device, weights_only=False))
fr_model.eval()

def get_feature(img_path):
    img = Image.open(img_path).convert('RGB')
    cropped = mtcnn(img)
    if cropped is None: return None
    tensor = cropped.unsqueeze(0).float().div_(255).sub_(0.5).div_(0.5).to(device)
    with torch.no_grad(): feat = fr_model(tensor)
    return torch.nn.functional.normalize(feat, p=2, dim=1)






# Main Experiment Loop 
# ...............................................................................
print("\n=== Starting Full Architecture Comparison ===")

for model_name, checkpoint_path in MODELS.items():
    print(f"\n{'='*50}\n▶ Testing Architecture: {model_name}\n{'='*50}")
    
    for dataset in DATASETS:
        morph_dir = os.path.join(BASE_DIR, dataset)
        if not os.path.exists(morph_dir):
            continue
        
        morph_files = sorted(glob.glob(os.path.join(morph_dir, "*.*")))[:3]
        
        for morph_path in morph_files:
            filename = os.path.basename(morph_path).replace('.jpg','').replace('.png','')
            print(f"\n[{model_name} | {dataset}] Processing: {filename}")
            
            # Extract Contexts
            id1 = filename.split('-vs-')[0]
            try:
                p_minus_path = glob.glob(os.path.join(BONAFIDE_DIR, f'{id1}*'))[0]
            except IndexError:
                continue
                
            feat_morph = get_feature(morph_path)
            feat_bona = get_feature(p_minus_path)
            
            if feat_morph is None or feat_bona is None:
                continue
                
            contexts = {'0': feat_morph.cpu().squeeze().numpy(), '1': feat_bona.cpu().squeeze().numpy()}
            os.makedirs(os.path.dirname(CONTEXT_SAVE_PATH), exist_ok=True)
            torch.save(contexts, CONTEXT_SAVE_PATH)
            
            # Save morph path for DDIM Inversion
            with open('/content/drive/MyDrive/HCML_Project/NegFaceDiff/current_morph_path.txt', 'w') as f:
                f.write(morph_path)
            
            for w in [0.0, 0.5, 1.0]:
                w_tag = f"w_{int(w*10)}"
                dst_dir = f"{BASE_OUTPUT}/{model_name}/{dataset}/{filename}/{w_tag}"
                os.makedirs(dst_dir, exist_ok=True)
                
                # Setup proper cache clearing based on architecture's checkpoint folder name
                cache_dir = f"./output/{os.path.basename(checkpoint_path)}/random_synthetic_uniform_1/rand/{w_tag}"
                os.system(f"rm -rf {cache_dir}")
                
                print(f"  [*] Generating w={w} ...")
                
                # Run sample.py (Removed silent mode for debugging)
                cmd = f"python sample.py sampling.method=rand sampling.n_contexts=1 create_contexts.n_contexts=1 sampling.n_samples_per_context=1 checkpoint.path={checkpoint_path} neg_prompt.w={w} sampling.ddim_step=200"
                # os.system(cmd)
                os.system(cmd + " > /dev/null 2>&1")
                
                src = f"{cache_dir}/samples/0.png"
                if os.path.exists(src):
                    os.system(f"cp {src} {dst_dir}/generated.png")
                    print(f"    [+] Saved successfully.")
                else:
                    print(f"    [-] Generation Failed for {w_tag}")

print("\n=== All Architecture Experiments Done! Check ./final_experiments_results ===")