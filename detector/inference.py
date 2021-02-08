"""
Load and pass images through the model here.

To see image of errors set img_check
    'NBC' if result should be no badminton court
    'BC' if result should be badmiton court
"""
import cv2
import os
import torch
import prep
from load import load_model

class Inference:
    """Evaluates model"""
    
    def __init__(self, model, processor):
        """Initialises image variables and loads model"""
        self.img_wid = 128
        self.img_hei = 72
        self.device, self.net = load_model(model, processor)
    
    def check(self, img, img_check=None):
        """Checks if image is a badminton court"""
        output = self.pass_(img)
        if output == 0:
            print(f'{img} No Badmiton Court')
        elif output == 1:
            print(f'{img} Badminton Court')
        if img_check:
            self.display_error(img, output, img_check)

    def check_dir(self, dir_, img_check=None):
        """Checks if a badminton court is in a directory"""
        for f in os.listdir(dir_):
            if '.jpg' in f or ".jpeg" in f:
                img = os.path.join(dir_, f)
                self.check(img, img_check)

    def check_num(self, dir_, start, stop, img_check=None):
        """Checks a range of numbered files for badminton court"""
        for f in range(start, stop):
            try:
                img = (f'{dir_}{f}.jpg')
                self.check(img, img_check)
            except cv2.error as e2:
                try:
                    path = (f'{dir_}{f}.jpeg')
                    self.check(img, img_check)
                except cv2.error as e:
                    pass

    def pass_(self, img):
        """Passes image array through model"""
        img_arr = prep.img(img, self.img_wid, self.img_hei)
        X = prep.arr(img_arr, self.img_wid, self.img_hei)
        
        with torch.no_grad():
            X = X.to(self.device)
            output = self.net(X)
            output = torch.argmax(output)
        return output

    def display_error(self, img, output, img_check):
        """Displays errors in directory for given classifer"""
        if img_check == 'NBC' and output == 1:
            print(f'Error should not be labelled badminton court')
            self.display_image(img)
        elif img_check == 'BC' and output == 0:
            print(f'Error should be labelled badminton court')
            self.display_image(img)

    def display_image(self, img):
        """Displays image"""
        display = cv2.imread(img)
        cv2.imshow(img, display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()