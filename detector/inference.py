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
import prep
from load_model import LoadModel


class Inference(LoadModel):
    """Evaluates model"""
    
    def __init__(self, model, processor):
        """Initialises image variables and loads model"""
        self.img_wid = 128
        self.img_hei = 72
        super().__init__(model, processor)
    
    def chk(self, ori_img, img_chk=None):
        """Checks if image is a badminton court"""
        if img_chk:
            img_chk = img_chk.upper()
        self.ori_img = ori_img
        self.pass_()
        if self.output == 0:
            print(f'{self.ori_img} No Badmiton Court')
        elif self.output == 1:
            print(f'{self.ori_img} Badminton Court')
        if img_chk:
            self.dis_err(img_chk)

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
        self.img, self.img_arr = prep.img(self.ori_img, self.img_wid, self.img_hei)
        self.X = prep.arr(self.img_arr, self.img_wid, self.img_hei)
        
        with torch.no_grad():
            self.X = self.X.to(self.device)
            self.output = self.net(self.X)
            self.output = torch.argmax(self.output)

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




