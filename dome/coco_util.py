import cv2
# import glob
# import natsort
import json
import numpy as np
# import os
# import magic
# import math

def project2D(joints, calib, imgwh=None, applyDistort=True):
    """
    Input:
    joints: N * 3 numpy array.
    calib: a dict containing 'R', 'K', 't', 'distCoef' (numpy array)

    Output:
    pt: 2 * N numpy array
    inside_img: (N, ) numpy array (bool)
    """
    x = np.dot(calib['R'], joints.T) + calib['t']
    xp = x[:2, :] / x[2, :]

    if applyDistort:
        X2 = xp[0, :] * xp[0, :]
        Y2 = xp[1, :] * xp[1, :]
        XY = X2 * Y2
        R2 = X2 + Y2
        R4 = R2 * R2
        R6 = R4 * R2

        dc = calib['distCoef']
        radial = 1.0 + dc[0] * R2 + dc[1] * R4 + dc[4] * R6
        tan_x = 2.0 * dc[2] * XY + dc[3] * (R2 + 2.0 * X2)
        tan_y = 2.0 * dc[3] * XY + dc[2] * (R2 + 2.0 * Y2)

        # xp = [radial;radial].*xp(1:2,:) + [tangential_x; tangential_y]
        xp[0, :] = radial * xp[0, :] + tan_x
        xp[1, :] = radial * xp[1, :] + tan_y

    # pt = bsxfun(@plus, cam.K(1:2,1:2)*xp, cam.K(1:2,3))';
    pt = np.dot(calib['K'][:2, :2], xp) + calib['K'][:2, 2].reshape((2, 1))

    if imgwh is not None:
        assert len(imgwh) == 2
        imw, imh = imgwh
        winside_img = np.logical_and(pt[0, :] > -0.5, pt[0, :] < imw-0.5) 
        hinside_img = np.logical_and(pt[1, :] > -0.5, pt[1, :] < imh-0.5) 
        inside_img = np.logical_and(winside_img, hinside_img) 
        inside_img = np.logical_and(inside_img, R2 < 1.0) 
        return pt, inside_img

    return pt

def load_json(path):
    if path.endswith(".json"):
        with open(path) as json_data:
            #print path
            d = json.load(json_data)
            json_data.close()
            return d

    print "Failed to Load JSON"
    stop
    return 0

def write_json(path, data):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)

def show_image(img, waitKeyValue = 1, name="win"):
    if img is not None:
        cv2.imshow(name,img)
        key = cv2.waitKey(waitKeyValue)
        if key == 32:
            while 1:
                key = cv2.waitKey(waitKeyValue)
                if key == 32:
                    break

def get_rect_from_points_only(points):
    minx = 1000000
    miny = 1000000
    maxx = 0
    maxy = 0
    for point in points:
        if len(point) == 3: 
            if point[2] == 0: continue
        if point[0] < minx:
            minx = point[0]
        if point[1] < miny:
            miny = point[1]
        if point[0] > maxx:
            maxx = point[0]
        if point[1] > maxy:
            maxy = point[1]
    return [int(minx), int(miny), int(maxx), int(maxy)]

def get_rect_from_points_only_bigger(points, imgageWidth, imgageHeight, ratio=0.1):
    rect = get_rect_from_points_only(points)
    rectW = rect[2]-rect[0]
    rectH = rect[3]-rect[1]
    rect[0] = int(max(0, rect[0]-ratio*rectW))
    rect[1] = int(max(0, rect[1]-ratio*rectH))
    rect[2] = int(min(imgageWidth-1, rect[2]+ratio*rectW))
    rect[3] = int(min(imgageHeight-1, rect[3]+ratio*rectH))
    return rect

# def mpiiann_to_np(cocoann):
#     total_size = len(cocoann)/3
#     # 0 not labeled (highly occluded or out of the image)
#     # 1 labeled but not visible
#     # 2 labeled and visible
#     # 3 not in dataset
#     nparr = np.zeros(shape=(total_size,3))
#     for i in range(0,total_size):
#         nparr[i,0] = cocoann[i*3 + 0]
#         nparr[i,1] = cocoann[i*3 + 1]
#         cocoann_prob =  cocoann[i*3 + 2]
#         if cocoann_prob == 1:
#             nparr[i,2] = 2
#         elif cocoann_prob == 0:
#             nparr[i,2] = 0
#         elif cocoann_prob == 3:
#             nparr[i,2] = 3
#     return nparr

# def l2(a,b):
#     return math.sqrt(pow(a[0]-b[0],2)+pow(a[1]-b[1],2))

# def convert_op(read_path, write_path, dataset_name, path=''):
#     sImageScale = 368.

#     with open(read_path) as json_data:
#         train_annotations = json.load(json_data)
#     numberAnnotations = len(train_annotations)

#     # Iterate through each image
#     counter = -1
#     jointAll = []
#     for i in range(0, numberAnnotations):
#         w = train_annotations[i]["annorect"][0]["img_width"]
#         h = train_annotations[i]["annorect"][0]["img_height"]
#         numerPeople = len(train_annotations[i]["annorect"]);

#         # Debug
#         img = None
#         #img = cv2.imread(path+train_annotations[i]["image_path"])

#         # Iterate through each image anno
#         previousCenters = [];
#         for p in range(0, numerPeople):
#             # Skip person if num parts too low or seg area too small
#             if train_annotations[i]["annorect"][p]["num_keypoints"] >= 5 and train_annotations[i]["annorect"][p]["area"] >= 32*32:
#                 # Skip person if distance to prev person is too small (So we combine them together)
#                 personCenter = [train_annotations[i]["annorect"][p]["bbox"][0] + train_annotations[i]["annorect"][p]["bbox"][2] / 2,
#                                 train_annotations[i]["annorect"][p]["bbox"][1] + train_annotations[i]["annorect"][p]["bbox"][3] / 2]
#                 addPerson = True;
#                 for k in range(0, len(previousCenters)):
#                     dist = l2([previousCenters[k][0], previousCenters[k][1]], personCenter)
#                     if dist < previousCenters[k][2]*0.3:
#                         addPerson = False
#                         break

#                 if img is not None:
#                     bbox = train_annotations[i]["annorect"][p]["bbox"]
#                     cv2.rectangle(img, (int(bbox[0]),int(bbox[1])),(int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])), 255, 2)
#                     show_image(img)
#                     time.sleep(1)

#                 # Add Person
#                 if addPerson:
#                     counter = counter + 1;
#                     data = dict()
#                     data["dataset"] = dataset_name
#                     data["img_paths"] = path + train_annotations[i]["image_path"]
#                     data["mask_paths"] = path + train_annotations[i]["image_path"]
#                     data["img_width"] = w
#                     data["img_height"] = h
#                     data["objpos"] = personCenter
#                     data["image_id"] = train_annotations[i]["image_id"]
#                     data["bbox"] = train_annotations[i]["annorect"][p]["bbox"]
#                     data["segment_area"] = train_annotations[i]["annorect"][p]["area"]
#                     data["num_keypoints"] = train_annotations[i]["annorect"][p]["num_keypoints"]
#                     data["num_keypoints"] = train_annotations[i]["annorect"][p]["num_keypoints"]
#                     data["joint_self"] = mpiiann_to_np(train_annotations[i]["annorect"][p]["keypoints"]).tolist()
#                     data["scale_provided"] = train_annotations[i]["annorect"][p]["bbox"][3] / sImageScale

#                     # Add other people in same image
#                     data["scale_provided_other"] = []
#                     data["objpos_other"] = []
#                     data["bbox_other"] = []
#                     data["segment_area_other"] = []
#                     data["num_keypoints_other"] = []
#                     data["joint_others"] = []
#                     counterOther = -1;
#                     for o in range(0, numerPeople):
#                         if o != p and train_annotations[i]["annorect"][o]["num_keypoints"] > 0:
#                             counterOther = counterOther + 1
#                             data["scale_provided_other"].append(train_annotations[i]["annorect"][o]["bbox"][3] / sImageScale)
#                             otherPersonCenter = [train_annotations[i]["annorect"][o]["bbox"][0] + train_annotations[i]["annorect"][o]["bbox"][2] / 2,
#                                 train_annotations[i]["annorect"][o]["bbox"][1] + train_annotations[i]["annorect"][o]["bbox"][3] / 2]
#                             data["objpos_other"].append(otherPersonCenter)
#                             data["bbox_other"].append(train_annotations[i]["annorect"][o]["bbox"])
#                             data["segment_area_other"].append(train_annotations[i]["annorect"][o]["area"])
#                             data["num_keypoints_other"].append(train_annotations[i]["annorect"][o]["num_keypoints"])
#                             data["joint_others"].append(mpiiann_to_np(train_annotations[i]["annorect"][o]["keypoints"]).tolist())
                    
#                     # Write Indexes
#                     data["annolist_index"] = i
#                     data["people_index"] = p
#                     data["numOtherPeople"] = len(data["joint_others"])

#                     # Update previous center
#                     previousCenters.append([personCenter[0], personCenter[1], max(train_annotations[i]["annorect"][p]["bbox"][2],train_annotations[i]["annorect"][p]["bbox"][3])])

#                     # Add to jointAll
#                     jointAll.append(data)

#     # Save to JSON
#     open(write_path, 'w').close()
#     with open(write_path, 'w') as outfile:
#         print "OP Annotations: " + str(len(jointAll))
#         json.dump(jointAll, outfile)
