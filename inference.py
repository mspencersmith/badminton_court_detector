import cv2
import os
import numpy as np
import torch
import time
from nn_model import Net

start = time.time()



class Inference:
    """Evaluates model"""
    
    def __init__(self, model, processor):
        """Initialises model and image variables"""
        self.img_wid = 128
        self.img_hei = 72
        self.model = model
        self.processor = processor
        self.load_model()
    
    def load_model(self):
        """Loads model on to GPU if availible otherwise loads to CPU"""
        
        if "cuda" in self.processor and torch.cuda.is_available==False():
            print("\nCuda unavailable\n")
            self.processor = "cpu"
        print(f"\nLoading neural network on {self.processor}..\n")
        self.device, self.net = to_dev()
        
    def _to_dev(self):
        """Initialises network onto device"""
        
        self.device = torch.device(self.processor)
        self.net = Net().to(self.device)
        self.net.load_state_dict(torch.load(self.model, map_location=device))
        self.net.eval()
        return self.device, self.net

    def show(self):
        """Shows if given image has a badminton court"""

        img = cv2.imread(self.model, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_wid, img_hei))
        img_arr = np.array(img)
        X = torch.Tensor(img_arr).view(-1, 1, img_wid, img_hei)
        X = X/255.0
        with torch.no_grad():
            X = X.to(self.device)
            output = net(X)
            output = torch.argmax(output)
            if output == 0:
                print(f'{self.model} Random')
            elif output == 1:
                print(f'{self.model} Badminton Court')
                cv2.imshow(self.model, img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        
inf = Inference('bcf_models/model-1608419488-lr1e-06-factor0.1pat2-thr0.01-val_pct0.2-bat_size150-128-72-512.pth', "cuda:0")


directory = "bcf_images/random/"
# directory = "bcf_images/badminton_court"
# directory = "bcf_images/overflow"

# show("bcf_images/random/0.jpg", device, net)

# for f in os.listdir(directory):
#     path = os.path.join(directory, f)
#     show(path, device, net)

# for f in range(0, 100):

#     try:
#         path = (f'{directory}{f}.jpeg')
#         show(path, device, net)
#     except cv2.error as e2:
#         pass
#     try:
#         path = (f'{directory}{f}.jpg')
#         show(path, device, net)
#     except cv2.error as e:
#         pass
