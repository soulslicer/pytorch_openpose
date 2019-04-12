from __future__ import print_function
import argparse
import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init 
import torchvision.models as models 
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import scipy.io as sio
import time
from collections import OrderedDict
import numpy as np
import torch.utils.model_zoo as model_zoo
import os 
import cv2
os.environ['GLOG_minloglevel'] = '2' 
#import caffe
dir_path = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, "/home/raaj/openpose_caffe_train/build/op/")
import opcaffe
import signal
exit = 0
def signal_handler(sig, frame):
    global exit
    exit = 1
signal.signal(signal.SIGINT, signal_handler)

from models import *

# Parsers
parser = argparse.ArgumentParser(description='OP')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--reload', action='store_true')
args = parser.parse_args()

# Setup Model
NAME = "weights"
model = Model(Body25("3x3"), ngpu=int(args.ngpu)).cuda()
model.train()

# Load weights etc.
iterations = 0
reload = int(args.reload)
if not reload:
    state = load_checkpoint(NAME)
    if state != None:
        iterations = state["iterations"]
        model.load_state_dict(state['state_dict'])
        print("Loaded Iteration " + str(iterations))

model.net.load_caffe()

params = {
    "batch_size" : 1,
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
    "source_background": "/media/raaj/Storage/openpose_train/dataset/lmdb_background",
    "normalization": 0,
    "add_distance": 0
}
myClass = opcaffe.OPCaffe(params)

# Loss
lr = 0.000100
parameters = [
        {"params": model.net.vgg19.parameters(), "lr": lr*1},
        {"params": model.net.pafA.parameters(), "lr": lr*4},
        {"params": model.net.pafB.parameters(), "lr": lr*4},
        {"params": model.net.hmA.parameters(), "lr": lr*4},
        {"params": model.net.hmB.parameters(), "lr": lr*4},
    ]
mseLoss = torch.nn.MSELoss()
optimizer = optim.Adam(parameters, lr=lr, betas=(0.9, 0.999))

# Iterate
while 1:
    iterations += 1
    batch = opcaffe.Batch()
    myClass.load(batch)

    # Split
    paf_mask = torch.tensor(batch.label[:, 0:TOTAL_PAFS]).cuda()
    hm_mask = torch.tensor(batch.label[:, TOTAL_PAFS:TOTAL_PAFS+TOTAL_HMS]).cuda()
    paf_truth = torch.tensor(batch.label[:, TOTAL_PAFS+TOTAL_HMS:TOTAL_PAFS+TOTAL_HMS+TOTAL_PAFS]).cuda()
    hm_truth = torch.tensor(batch.label[:, TOTAL_PAFS+TOTAL_HMS+TOTAL_PAFS:TOTAL_PAFS+TOTAL_HMS+TOTAL_PAFS+TOTAL_HMS]).cuda()
    imgs = torch.tensor(batch.data).cuda()

    # Mask
    paf_truth_m = torch.mul(paf_truth, paf_mask)
    hm_truth_m = torch.mul(hm_truth, hm_mask)

    # Forward Model
    pafs_pred, hms_pred = model.forward(imgs)

    # Multiply with Masks
    loss = 0
    for i in range(0, ITERATIONS):
        paf_pred_m = torch.mul(pafs_pred[i], paf_mask)
        hm_pred_m = torch.mul(hms_pred[i], hm_mask)
        loss += mseLoss(paf_pred_m, paf_truth_m)
        loss += mseLoss(hm_pred_m, hm_truth_m)

    # Opt
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save
    if iterations % 20 == 0 or exit:
        print("Saving")
        save_checkpoint({
            'iterations': iterations,
            'state_dict': model.state_dict(),
        }, NAME)
    if exit: sys.exit()


    print((iterations,loss))


    img_viz = imgs.detach().cpu().numpy().copy()[0,0,:,:]
    hm_pred_viz = hms_pred[ITERATIONS-1].detach().cpu().numpy().copy()[0,0,:,:]
    hm_truth_viz = hm_truth_m.cpu().numpy().copy()[0,0,:,:]
    cv2.imshow("hm_pred_viz", cv2.resize(hm_pred_viz, (0,0), fx=8, fy=8, interpolation = cv2.INTER_CUBIC))
    cv2.imshow("hm_truth_viz", cv2.resize(hm_truth_viz, (0,0), fx=8, fy=8, interpolation = cv2.INTER_CUBIC))
    cv2.imshow("img", img_viz+0.5)
    cv2.waitKey(-1)

    # print(pafs_truth[0].shape)
    # print(hms_truth[0].shape)


    # print(batch.label.shape)
    # print(batch.data.shape)






# # Xavier Init

# empty_image = torch.tensor(np.zeros((1,3,656,368), dtype=np.float32)).cuda()
# model.forward(empty_image)

# for i in range(0,5):
#     torch.cuda.synchronize()
#     start_time = time.time()
#     model.forward(empty_image)
#     torch.cuda.synchronize()
#     elapsed_time = time.time() - start_time

#     print(elapsed_time)

# stop

# ######

# # Load Caffe Try?
# net = caffe.Net('/media/raaj/Storage/openpose_train/training_results_gines_new/pose_image/pose_deploy.prototxt',
#                 '/media/raaj/Storage/openpose_train/training_results_gines_new/pose_image/model/pose_iter_608000.caffemodel',
#                 caffe.TEST)

# weights_load = {}
# for key in net.params:
#     print(key)
#     W = net.params[key][0].data[...]
#     weights_load["vgg19"+"."+key+"."+"weight"] = torch.tensor(W)
#     if len(net.params[key]) > 1:
#         b = net.params[key][1].data[...]
#         weights_load["vgg19"+"."+key+"."+"bias"] = torch.tensor(b)
#     if key == "conv4_2": break
# # stop

# # # Load VGG19
# # vgg19_model_url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
# # vgg_state_dict = model_zoo.load_url(vgg19_model_url, model_dir=dir_path)  # save to local path
# # vgg_keys = vgg_state_dict.keys()
# # weights_load = {}
# # for i in range(2):  # use first 10 conv layers of vgg19, 20 weights in all (weight+bias)
# #     weights_load[list(vgg19.state_dict().keys())[i]] = vgg_state_dict[list(vgg_keys)[i]]

# # print(weights_load.keys())
# # print(weights_load["vgg19.conv1_1.bias"].dtype)
# # print(weights_load["vgg19.conv1_1.weight"].dtype)
# # stop

# state = model.state_dict()
# state.update(weights_load)
# model.load_state_dict(state)