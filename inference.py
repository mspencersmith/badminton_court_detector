"""
Load and pass images through the model here.

To see image of errors set img_check
    'NBC' if result should be no badminton court
    'BC' if result should be badmiton court
"""
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
        if "cuda" in self.processor and torch.cuda.is_available()==False:
            print("\nCuda unavailable\n")
            self.processor = "cpu"
        print(f"\nLoading neural network on {self.processor}..\n")
        self.device, self.net = self._to_dev()
        
    def _to_dev(self):
        """Initialises model onto device"""
        self.device = torch.device(self.processor)
        self.net = Net().to(self.device)
        self.net.load_state_dict(torch.load(self.model, map_location=self.device))
        self.net.eval()
        return self.device, self.net
    
    def chk(self, ori_img, img_chk=None):
        """Checks if image is a badminton court"""
        img_chk = img_chk.upper()
        self.ori_img = ori_img
        self.pass_()
        if self.output == 0:
            print(f'{self.ori_img} No Badmiton Court')
        elif self.output == 1:
            print(f'{self.ori_img} Badminton Court')
        if img_chk:
            self.dis_err(img_chk)
        return self.output

    def chk_dir(self, dir_, img_chk=None):
        """Checks if a badminton court is in a directory"""
        for f in os.listdir(dir_):
            if '.jpg' in f or ".jpeg" in f:
                self.ori_img = os.path.join(dir_, f)
                self.chk(self.ori_img, img_chk)

    def chk_num(self, dir_, start, stop, img_chk=None):
        """Checks a range of numbered files for badminton court"""
        for f in range(start, stop):
            try:
                self.ori_img = (f'{dir_}{f}.jpg')
                self.chk(self.ori_img, img_chk)
            except cv2.error as e2:
                try:
                    path = (f'{dir_}{f}.jpeg')
                    self.chk(self.ori_img, img_chk)
                except cv2.error as e:
                    pass

    def pass_(self):
        """Passes image array through model"""
        self.prep_img()
        self.prep_arr()
        
        with torch.no_grad():
            self.X = self.X.to(self.device)
            self.output = self.net(self.X)
            self.output = torch.argmax(self.output)
            # print(self.output)
            return self.output

    def prep_img(self, ori_img=None):
        """Converts image to array"""
        if ori_img:
            self.ori_img = ori_img
        self.img = cv2.imread(self.ori_img, cv2.IMREAD_GRAYSCALE)
        self.img = cv2.resize(self.img, (self.img_wid, self.img_hei))
        self.img_arr = np.array(self.img)
        return self.img_arr
        
    def prep_arr(self, arr=None):
        """Prepares array for pass through model"""
        if arr:
            self.img_arr = arr
        self.X = torch.Tensor(self.img_arr).view(-1, 1, self.img_wid, self.img_hei)
        self.X = self.X/255.0
        return self.X

    def dis_err(self, img_chk):
        """Displays errors in directory for given classifer"""
        if img_chk == 'NBC' and self.output == 1:
            print(f'Error should not be labelled badminton court')
            self._dis_img()
        elif img_chk == 'BC' and self.output == 0:
            print(f'Error should be labelled badminton court')
            self._dis_img()

    def _dis_img(self):
        """Displays image"""
        img = cv2.imread(self.ori_img)
        cv2.imshow(self.ori_img, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




