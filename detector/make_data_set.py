"""
Create data set here. Convert images to grayscale so there are less 
channels which takes up less memory. Also resize images so they are all the 
same size and take up less memory" 

"""

import os
import prep
import numpy as np
from pathlib import Path
from tqdm import tqdm


class Data():
    """Creates data set of images and their labels"""
    BASE_DIR = Path(__file__).resolve().parent.parent
    img_wid = 128
    img_hei = 72
    ran = os.path.join(BASE_DIR, "data/images/random")
    bc = os.path.join(BASE_DIR, "data/images/badminton_court")
    output = os.path.join(BASE_DIR, "data/data_set.npy")
    LABELS = {ran: 0, bc: 1}
    data_set = []
    
    ran_count = 0
    bc_count = 0

    def make_data_set(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    _, img_arr = prep.img(path, self.img_wid, self.img_hei)
                    self.data_set.append([img_arr, np.eye(2)
                        [self.LABELS[label]]])

                    if label == self.ran:
                        self.ran_count += 1
                    elif label == self.bc:
                        self.bc_count += 1

                except Exception as e:
                    print(label. f, str(e))

        np.random.shuffle(self.data_set)
        np.save(self.output, self.data_set)
        print("Random: ", self.ran_count)
        print("Badminton Court: ", self.bc_count)
        print(f"Total: {len(self.data_set)}")


data = Data()
data.make_data_set()