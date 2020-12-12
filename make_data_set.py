import os
import cv2
import numpy as np
from tqdm import tqdm


class Data():
    img_wid = 128
    img_hei = 72
    ran = "bcf_images/random"
    bc = "bcf_images/badminton_court"
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
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.img_wid, self.img_hei))
                    self.data_set.append([np.array(img), np.eye(2)
                        [self.LABELS[label]]])
                    # print(np.eye(2)[self.LABELS[label]])
                    # print(self.data_set[0])

                    if label == self.ran:
                        self.ran_count += 1
                    elif label == self.bc:
                        self.bc_count += 1

                except Exception as e:
                    # pass
                    print(label. f, str(e))

        np.random.shuffle(self.data_set)
        np.save("bcf_data/data_set.npy", self.data_set)
        print("Random: ", self.ran_count)
        print("Badminton Court: ", self.bc_count)
        print(f"Total: {len(self.data_set)}")


data = Data()
data.make_data_set()