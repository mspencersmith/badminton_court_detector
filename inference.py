import cv2
import os
import numpy as np
import torch
import time
from nn_model import Net

start = time.time()

def load_model(file):
    """Loads model on to GPU if availible otherwise loads to CPU"""
    
    if torch.cuda.is_available():
        print(f"\nLoading neural network on GPU..\n")
        device = torch.device("cuda:0")
        net = Net().to(device)
        net.load_state_dict(torch.load(file, map_location=device))
        net.eval()
    else:
        print(f"\nLoading neural network on CPU..\n")
        device = torch.device("cpu")
        net = Net().to(device)
        net.load_state_dict(torch.load(file, map_location=device))
        net.eval()
    return device, net

def show(file, device, net):
    """Detects if given image has a badminton court"""

    img_wid = 128
    img_hei = 72
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_wid, img_hei))
    img_arr = np.array(img)
    X = torch.Tensor(img_arr).view(-1, 1, img_wid, img_hei)
    X = X/255.0
    with torch.no_grad():
        X = X.to(device)
        output = net(X)
        output = torch.argmax(output)
        return output

device, net = load_model('bcf_models/model-1608175256-lr1e-06-factor0.1pat2-thr0.01-val_pct0.2-bat_size150-128-72-512.pth')

finish = time.time()
mins = round((finish - start)/60, 2)
print(f"\nTotal time taken {mins} minutes.")

directory = "badminton_court_finder/bcf_images/random"
# directory = "badminton_court_finder/bcf_images/badminton_court"
# directory = "badminton_court_finder/bcf_images/overflow"

# show("badminton_court_finder/bcf_images/overflow/0.jpg", device, net)

# for f in os.listdir(directory):
#     path = os.path.join(directory, f)
#     show(path, device, net)

for f in range(10848, 11000):
    try:
        path = (f'{directory}{f}.jpg')
        show(path, device, net)
    except cv2.error as e:
        raise e
    else:
        path = (f'{directory}{f}.jpeg')
        show(path, device, net)

