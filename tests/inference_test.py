import os.path
from pathlib import Path
from inference import Inference

BASE_DIR = Path(__file__).resolve().parent.parent

models = os.path.join(BASE_DIR, "models")
data = os.path.join(BASE_DIR, "data")

inf = Inference(f"{models}/model-1612398660-lr1e-06-factor0.1pat2-thr0.01-val_pct0.2-train_batch150-128-72-512-noSofm-Relu-mse.pth", "cuda:0")
inf.check(f"{data}/images/random/1.jpg") # Checks single image
inf.check_dir(f"{data}/images/overflow/", 'nbc') # Checks directory
inf.check_num(f"{data}/images/badminton_court/", 0, 1000, 'BC') # Checks directory where files are numbered