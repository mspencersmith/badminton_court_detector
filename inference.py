import cv2
import os
import numpy as np
import torch
import time
from nn_model import Net

start = time.time()

img_wid = 128
img_hei = 72

def load_model(file):
    """Loads model on to GPU if availible otherwise loads to CPU"""
    
    if torch.cuda.is_available():
        print(f"\nLoading neural network on GPU..\n")
        device, net = to_dev(file, "cuda:0")
    else:
        print(f"\nLoading neural network on CPU..\n")
        device, net = to_dev(file, "cpu")

    return device, net

def to_dev(file, processor):
    """Transfers model to device"""
    
    device = torch.device(processor)
    net = Net().to(device)
    net.load_state_dict(torch.load(file, map_location=device))
    net.eval()
    return device, net

def show(file, device, net):
    """Shows if given image has a badminton court"""

    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_wid, img_hei))
    img_arr = np.array(img)
    X = torch.Tensor(img_arr).view(-1, 1, img_wid, img_hei)
    X = X/255.0
    with torch.no_grad():
        X = X.to(device)
        output = net(X)
        output = torch.argmax(output)
        if output == 0:
            print(f'{file} Random')
        elif output == 1:
            print(f'{file} Badminton Court')
            cv2.imshow(file, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        

device, net = load_model('bcf_models/model-1608419488-lr1e-06-factor0.1pat2-thr0.01-val_pct0.2-bat_size150-128-72-512.pth')

finish = time.time()
mins = round((finish - start)/60, 2)
print(f"\nTotal time taken {mins} minutes.")

directory = "bcf_images/random/"
# directory = "bcf_images/badminton_court"
# directory = "bcf_images/overflow"

# show("bcf_images/random/0.jpg", device, net)

# for f in os.listdir(directory):
#     path = os.path.join(directory, f)
#     show(path, device, net)

for f in range(0, 100):

    try:
        path = (f'{directory}{f}.jpeg')
        show(path, device, net)
    except cv2.error as e2:
        pass
    try:
        path = (f'{directory}{f}.jpg')
        show(path, device, net)
    except cv2.error as e:
        pass
