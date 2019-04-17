from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import random
import cv2
import time
import PIL
import skimage
import sys
sys.path.insert(0, "/home/raaj/openpose_caffe_train/build/op/")
import opcaffe
from threading import Thread, Lock
mutex = Lock()


class OPLoader(torch.utils.data.Dataset):
    def __init__(self, WORKER_SIZE, BATCH_SIZE):
        self.WORKER_SIZE = WORKER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        if BATCH_SIZE % WORKER_SIZE != 0:
            print("Invalid sizes")
            stop

        self.workers = []
        for i in range(0, self.WORKER_SIZE):
            params = {
                "batch_size" : self.BATCH_SIZE/self.WORKER_SIZE,
                "stride": 8,
                "max_degree_rotations": "45.0",
                "crop_size_x": 368,
                "crop_size_y": 368,
                "center_perterb_max": 40.0,
                "center_swap_prob": 0.0,
                "scale_prob": 1.0,
                "scale_mins": "0.333333333333",
                "scale_maxs": "1.5",
                "target_dist": 0.600000023842,
                "number_max_occlusions": "2",
                "sigmas": "7.0",
                "models": "COCO_25B_23;COCO_25B_17;MPII_25B_16;PT_25B_15",
                "sources": "/media/raaj/Storage/openpose_train/dataset/lmdb_coco2017_foot;/media/raaj/Storage/openpose_train/dataset/lmdb_coco;/media/raaj/Storage/openpose_train/dataset/lmdb_mpii;/media/raaj/Storage/openpose_train/dataset/lmdb_pt2_train",
                "probabilities": "0.05;0.85;0.05;0.05",
                #"probabilities": "0.5;0.5;0.0;0.0",
                "source_background": "/media/raaj/Storage/openpose_train/dataset/lmdb_background",
                "normalization": 0,
                "add_distance": 0,
                "msize": self.WORKER_SIZE,
                "mrank": i
            }
            self.workers.append(opcaffe.OPCaffe(params))

    def __getitem__(self, index):
        pid = os.getpid() % self.WORKER_SIZE
        batch = opcaffe.Batch()
        print("call")
        self.workers[pid].load(batch)
        return torch.tensor(batch.data), torch.tensor(batch.label)

    def __len__(self):
        return 13605+122542+26961+70316


if __name__ == "__main__":

    WORKER_SIZE = 2
    BATCH_SIZE = 20

    kwargs = {'num_workers': WORKER_SIZE, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        OPLoader(WORKER_SIZE, BATCH_SIZE),
        batch_size=WORKER_SIZE, shuffle=False, **kwargs)

    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.flatten(0,1)
        label = label.flatten(0,1)

        time.sleep(2)
        print("BatchIDX: " + str(batch_idx), data.shape, label.shape)

        for i in range(0, data.shape[0]):
            img_viz = data.detach().cpu().numpy().copy()[i,0,:,:]
            cv2.putText(img_viz,str(batch_idx), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            cv2.imshow("img", img_viz+0.5)
            cv2.waitKey(15)
            break

        #break


        pass