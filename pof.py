import numpy as np
import numpy.linalg as nl
import pickle
import json
import os

IMAGE_ROOT = '/media/posefs0c/panopticdb/'
TOTALPOSE_ROOT = '/media/posefs0c/Users/donglaix/Experiments/totalPose/'
OP_CAFFE_TRAIN_PATH = '/home/raaj/openpose_caffe_train/build/op/'
LMDB_BACKGROUND = "/media/raaj/Storage/openpose_train/dataset/lmdb_background"

# {0,  "Nose"}, # 1
# {1,  "LEye"}, # 15
# {2,  "REye"}, # 17
# {3,  "LEar"}, # 16
# {4,  "REar"}, # 18
# {5,  "LShoulder"}, # 3
# {6,  "RShoulder"}, # 9
# {7,  "LElbow"}, # 4
# {8,  "RElbow"}, # 10
# {9,  "LWrist"}, # 5
# {10, "RWrist"}, # 11
# {11, "LHip"}, # 6
# {12, "RHip"}, # 12
# {13, "LKnee"}, # 7
# {14, "RKnee"}, # 13
# {15, "LAnkle"}, # 8
# {16, "RAnkle"}, # 14
# {17, "UpperNeck"},
# {18, "HeadTop"},
# {19, "LBigToe"},
# {20, "LSmallToe"},
# {21, "LHeel"},
# {22, "RBigToe"},
# {23, "RSmallToe"},
# {24, "RHeel"},

dome_to_body25b = dict()
dome_to_body25b[0] = 1
dome_to_body25b[1] = 15
dome_to_body25b[2] = 17
dome_to_body25b[3] = 16
dome_to_body25b[4] = 18
dome_to_body25b[5] = 3
dome_to_body25b[6] = 9
dome_to_body25b[7] = 4
dome_to_body25b[8] = 10
dome_to_body25b[9] = 5
dome_to_body25b[10] = 11
dome_to_body25b[11] = 6
dome_to_body25b[12] = 12
dome_to_body25b[13] = 7
dome_to_body25b[14] = 13
dome_to_body25b[15] = 8
dome_to_body25b[16] = 14

def convert(points):
    new_points = np.zeros((17,3))

    for i in range(0, new_points.shape[0]):

        try:
            new_points[i, :] = points[dome_to_body25b[i], :]
        except:
            new_points[i, :] = np.array([50000,50000,0.0001])

    return new_points

class DomeReader():

    def __init__(self, mode='training', objtype=0, shuffle=False, batch_size=1, crop_noise=False, full_only=True, head_top=True, debug=False, savename="human.pkl"):
        self.image_root = IMAGE_ROOT
        self.totalpose_root = TOTALPOSE_ROOT

        # read data from a4
        path_to_db = self.totalpose_root + '/data/a4_collected_p2.pkl'
        path_to_calib = self.totalpose_root + '/data/camera_data_a4_p2.pkl'

        print("loading..")
        with open(path_to_db, 'rb') as f:
            db_data = pickle.load(f)

            # # Writing
            # pickle.dump(db_data, open(self.totalpose_root + '/data/a4_collected_p2.pkl',"wb"), protocol=2)

        print("loading..")
        with open(self.totalpose_root + '/data/a4_hands_annotated.txt') as f:
            hand_annots = {}
            for line in f:
                strs = line.split()
                hand_annots[tuple(strs[:3])] = eval(strs[3])

        if mode == 'training':
            mode_data = db_data['training_data']
        else:
            mode_data = db_data['testing_data']

        print("loading..")
        with open(path_to_calib, 'rb') as f:
            calib_data = pickle.load(f)

            # # Writing
            # pickle.dump(calib_data, open(self.totalpose_root + '/data/camera_data_a4_p2.pkl',"wb"), protocol=2)

        human3d = {'body': [], 'left_hand': [], 'right_hand': [], 'body_valid': [], 'left_hand_valid': [], 'right_hand_valid': []}
        calib = {'K': [], 'R': [], 't': [], 'distCoef': []}
        img_dirs = []

        counter = 0
        for data3d in mode_data:
            counter+=1
            if counter == 5 and debug: break
            print(float(counter)/float(len(mode_data)))
            seqName = data3d['seqName']
            frame_str = data3d['frame_str']

            # check for manual annotation, remove the annotation if the hand is annotated as incorrect.
            if 'left_hand' in data3d and not hand_annots[(seqName, frame_str, 'left')]:
                del data3d['left_hand']
            if 'right_hand' in data3d and not hand_annots[(seqName, frame_str, 'righ')]:
                del data3d['right_hand']

            if objtype == 0:
                body3d = np.array(data3d['body']['landmarks'], dtype=np.float32).reshape(-1, 3)
                #body3d = convert(body3d )

            elif objtype == 1:
                # left hand or right hand must be valid
                if 'left_hand' in data3d:
                    left_hand3d = np.array(data3d['left_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                if 'right_hand' in data3d:
                    right_hand3d = np.array(data3d['right_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                if ('left_hand' not in data3d) and ('right_hand' not in data3d):
                    continue

            else:
                assert objtype == 2
                body3d = np.array(data3d['body']['landmarks'], dtype=np.float32).reshape(-1, 3)
                # both left and right hand must be valid
                if 'left_hand' in data3d:
                    left_hand3d = np.array(data3d['left_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                    # discard the sample if hand is wanted but there is no left hand.
                else:
                    continue
                if 'right_hand' in data3d:
                    right_hand3d = np.array(data3d['right_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                else:
                    continue

            if objtype == 0:
                for camIdx, camDict in data3d['body']['2D'].items():
                    if full_only:
                        cond_inside = all(camDict['insideImg'])
                    else:  # if not full_only, use the image if at least half keypoints are visible
                        inside_ratio = np.float(np.sum(camDict['insideImg'])) / len(camDict['insideImg'])
                        cond_inside = (inside_ratio > 0.5)
                    if any(camDict['occluded']) or not cond_inside:
                        continue
                    human3d['body'].append(body3d)
                    human3d['body_valid'].append(np.ones((20 if head_top else 19,), dtype=bool))
                    calib['K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                    calib['R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                    calib['t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                    calib['distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                    img_dirs.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a4', seqName, frame_str, camIdx, frame_str))

            elif objtype == 1:
                if 'left_hand' in data3d:
                    for camIdx, camDict in data3d['left_hand']['2D'].items():
                        if any(data3d['left_hand']['2D'][camIdx]['occluded']) or not all(data3d['left_hand']['2D'][camIdx]['insideImg']) or data3d['left_hand']['2D'][camIdx]['overlap']:
                            continue
                        human3d['left_hand'].append(left_hand3d)
                        human3d['right_hand'].append(np.zeros((21, 3), dtype=np.float32))
                        human3d['left_hand_valid'].append(np.ones((21,), dtype=bool))
                        human3d['right_hand_valid'].append(np.zeros((21,), dtype=bool))
                        calib['K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                        calib['R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                        calib['t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                        calib['distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                        img_dirs.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a4', seqName, frame_str, camIdx, frame_str))

                if 'right_hand' in data3d:
                    for camIdx, camDict in data3d['right_hand']['2D'].items():
                        if any(data3d['right_hand']['2D'][camIdx]['occluded']) or not all(data3d['right_hand']['2D'][camIdx]['insideImg']) or data3d['right_hand']['2D'][camIdx]['overlap']:
                            continue
                        human3d['right_hand'].append(right_hand3d)
                        human3d['left_hand'].append(np.zeros((21, 3), dtype=np.float32))
                        human3d['left_hand_valid'].append(np.zeros((21,), dtype=bool))
                        human3d['right_hand_valid'].append(np.ones((21,), dtype=bool))
                        calib['K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                        calib['R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                        calib['t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                        calib['distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                        img_dirs.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a4', seqName, frame_str, camIdx, frame_str))

            else:
                assert objtype == 2
                for camIdx, camDict in data3d['body']['2D'].items():
                    if any(camDict['occluded']) or not all(camDict['insideImg']):
                        continue
                    if any(data3d['left_hand']['2D'][camIdx]['occluded']) or not all(data3d['left_hand']['2D'][camIdx]['insideImg']):
                        continue
                    if any(data3d['right_hand']['2D'][camIdx]['occluded']) or not all(data3d['right_hand']['2D'][camIdx]['insideImg']):
                        continue
                    # If this line is reached, the sample and cam view is valid.
                    human3d['body'].append(body3d)
                    human3d['left_hand'].append(left_hand3d)
                    human3d['right_hand'].append(right_hand3d)
                    human3d['body_valid'].append(np.ones((18,), dtype=bool))
                    human3d['left_hand_valid'].append(np.ones((21,), dtype=bool))
                    human3d['right_hand_valid'].append(np.ones((21,), dtype=bool))
                    calib['K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                    calib['R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                    calib['t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                    calib['distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                    img_dirs.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a4', seqName, frame_str, camIdx, frame_str))

        if mode == 'evaluation':
            if objtype == 2:
                openpose_output_file = '/home/donglaix/Documents/Experiments/dome_valid/a4_openpose.json'
                assert os.path.exists(openpose_output_file)
                with open(openpose_output_file) as f:
                    openpose_data = json.load(f)
                openpose_data = np.array(openpose_data, dtype=np.float32).reshape(-1, 70, 3)
                openpose_valid = (openpose_data[:, :, 2] >= 0.5)
                openpose_data[:, :, 0] *= openpose_valid
                openpose_data[:, :, 1] *= openpose_valid
                openpose_face = openpose_data[:, :, :2]
                human3d['openpose_face'] = openpose_face

        # read data from a5
        path_to_db = self.totalpose_root + '/data/a5_collected_p2.pkl'
        path_to_calib = self.totalpose_root + '/data/camera_data_a5_p2.pkl'

        with open(path_to_db, 'rb') as f:
            db_data = pickle.load(f)

            # # Writing
            # pickle.dump(db_data, open(self.totalpose_root + '/data/a5_collected_p2.pkl',"wb"), protocol=2)

        if mode == 'training':
            mode_data = db_data['training_data']
        else:
            mode_data = db_data['testing_data']

        with open(path_to_calib, 'rb') as f:
            calib_data = pickle.load(f)

            # # Writing
            # pickle.dump(calib_data, open(self.totalpose_root + '/data/camera_data_a5_p2.pkl',"wb"), protocol=2)

        counter = 0
        for data3d in mode_data:
            counter+=1
            if counter == 5 and debug: break
            print(float(counter)/float(len(mode_data)))
            seqName = data3d['seqName']
            frame_str = data3d['frame_str']

            if objtype == 0:
                body3d = np.array(data3d['body']['landmarks'], dtype=np.float32).reshape(-1, 3)
                #body3d = convert(body3d)

            elif objtype == 1:
                # left hand or right hand must be valid
                if 'left_hand' in data3d:
                    left_hand3d = np.array(data3d['left_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                if 'right_hand' in data3d:
                    right_hand3d = np.array(data3d['right_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                if ('left_hand' not in data3d) and ('right_hand' not in data3d):
                    continue

            else:
                assert objtype == 2
                body3d = np.array(data3d['body']['landmarks'], dtype=np.float32).reshape(-1, 3)
                # both left and right hand must be valid
                if 'left_hand' in data3d:
                    left_hand3d = np.array(data3d['left_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                    # discard the sample if hand is wanted but there is no left hand.
                else:
                    continue
                if 'right_hand' in data3d:
                    right_hand3d = np.array(data3d['right_hand']['landmarks'], dtype=np.float32).reshape(-1, 3)
                else:
                    continue

            if objtype == 0:
                for camIdx, camDict in data3d['body']['2D'].items():
                    if full_only:
                        cond_inside = all(camDict['insideImg'])
                    else:  # if not full_only, use the image if at least half keypoints are visible
                        inside_ratio = np.float(np.sum(camDict['insideImg'])) / len(camDict['insideImg'])
                        cond_inside = (inside_ratio > 0.5)
                    if any(camDict['occluded']) or not cond_inside:
                        continue
                    human3d['body'].append(body3d)
                    human3d['body_valid'].append(np.ones((20 if head_top else 19,), dtype=bool))
                    calib['K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                    calib['R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                    calib['t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                    calib['distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                    img_dirs.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a5', seqName, frame_str, camIdx, frame_str))

            elif objtype == 1:
                if 'left_hand' in data3d:
                    for camIdx, camDict in data3d['left_hand']['2D'].items():
                        if any(data3d['left_hand']['2D'][camIdx]['occluded']) or not all(data3d['left_hand']['2D'][camIdx]['insideImg']) or data3d['left_hand']['2D'][camIdx]['overlap']:
                            continue
                        human3d['left_hand'].append(left_hand3d)
                        human3d['right_hand'].append(np.zeros((21, 3), dtype=np.float32))
                        human3d['left_hand_valid'].append(np.ones((21,), dtype=bool))
                        human3d['right_hand_valid'].append(np.zeros((21,), dtype=bool))
                        calib['K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                        calib['R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                        calib['t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                        calib['distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                        img_dirs.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a5', seqName, frame_str, camIdx, frame_str))

                if 'right_hand' in data3d:
                    for camIdx, camDict in data3d['right_hand']['2D'].items():
                        if any(data3d['right_hand']['2D'][camIdx]['occluded']) or not all(data3d['right_hand']['2D'][camIdx]['insideImg']) or data3d['right_hand']['2D'][camIdx]['overlap']:
                            continue
                        human3d['right_hand'].append(right_hand3d)
                        human3d['left_hand'].append(np.zeros((21, 3), dtype=np.float32))
                        human3d['left_hand_valid'].append(np.zeros((21,), dtype=bool))
                        human3d['right_hand_valid'].append(np.ones((21,), dtype=bool))
                        calib['K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                        calib['R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                        calib['t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                        calib['distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                        img_dirs.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a5', seqName, frame_str, camIdx, frame_str))

            else:
                assert objtype == 2
                for camIdx, camDict in data3d['body']['2D'].items():
                    if any(camDict['occluded']) or not all(camDict['insideImg']):
                        continue
                    if any(data3d['left_hand']['2D'][camIdx]['occluded']) or not all(data3d['left_hand']['2D'][camIdx]['insideImg']):
                        continue
                    if any(data3d['right_hand']['2D'][camIdx]['occluded']) or not all(data3d['right_hand']['2D'][camIdx]['insideImg']):
                        continue
                    # If this line is reached, the sample and cam view is valid.
                    human3d['body'].append(body3d)
                    human3d['left_hand'].append(left_hand3d)
                    human3d['right_hand'].append(right_hand3d)
                    human3d['body_valid'].append(np.ones((18,), dtype=bool))
                    human3d['left_hand_valid'].append(np.ones((21,), dtype=bool))
                    human3d['right_hand_valid'].append(np.ones((21,), dtype=bool))
                    calib['K'].append(calib_data[seqName][camIdx]['K'].astype(np.float32))
                    calib['R'].append(calib_data[seqName][camIdx]['R'].astype(np.float32))
                    calib['t'].append(calib_data[seqName][camIdx]['t'][:, 0].astype(np.float32))
                    calib['distCoef'].append(calib_data[seqName][camIdx]['distCoef'].astype(np.float32))
                    img_dirs.append('{}/{}/hdImgs/{}/{}/00_{:02d}_{}.jpg'.format(self.image_root, 'a5', seqName, frame_str, camIdx, frame_str))

        human3d.update(calib)
        human3d['img_dirs'] = img_dirs

        pickle.dump(human3d, open(savename,"wb"), protocol=2)

    def get(self, withPAF=True):
        d = super(DomeReader, self).get(withPAF=withPAF)
        return d

import cv2
def project2D(joints, calib, imgwh=None, applyDistort=True):
    """
    Input:
    joints: N * 3 numpy array.
    calib: a dict containing 'R', 'K', 't', 'distCoef' (numpy array)

    Output:
    pt: 2 * N numpy array
    inside_img: (N, ) numpy array (bool)
    """
    calib['t'] = calib['t'].reshape((3,1))
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

    # Image Check
    assert len(imgwh) == 2
    imw, imh = imgwh
    winside_img = np.logical_and(pt[0, :] > -0.5, pt[0, :] < imw-0.5) 
    hinside_img = np.logical_and(pt[1, :] > -0.5, pt[1, :] < imh-0.5) 
    inside_img = np.logical_and(winside_img, hinside_img) 
    inside_img = np.logical_and(inside_img, R2 < 1.0) 

    return pt, inside_img, x


def get_rect(points):
    minx = 1000000
    miny = 1000000
    maxx = 0
    maxy = 0
    for point in points:
        if point[2] == 2: continue
        if point[0] < minx:
            minx = point[0]
        if point[1] < miny:
            miny = point[1]
        if point[0] > maxx:
            maxx = point[0]
        if point[1] > maxy:
            maxy = point[1]
    return [int(minx), int(miny), int(maxx), int(maxy)]

def getL2Dist(a, b):
    return np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)

def getVector(a,b, norm = True):
    if norm == False:
        return (b[0]-a[0], b[1]-a[1])
    else:
        vec = (b[0]-a[0], b[1]-a[1])
        n = np.sqrt(vec[0]**2 + vec[1]**2)
        vec = (vec[0]/n, vec[1]/n)
        return vec

def normalize(a):
    mag = np.sqrt(a[0]**2 + a[1]**2)
    return (a[0]/mag, a[1]/mag)

def dotProd(a,b, normalize = True):
    normalDot = a[0]*b[0] + a[1]*b[1]
    if not normalize:
        return normalDot
    aMag = np.sqrt(a[0]**2 + a[1]**2)
    bMag = np.sqrt(b[0]**2 + b[1]**2)
    return normalDot/(aMag*bMag)

def samplePoints(A, B, sample_dist = 20, sample_min = 5):
    l2Dist = getL2Dist(A,B)
    vec_nonorm = getVector(A, B, False)
    sample_points = []
    prev_sp = None
    if l2Dist == 0:
        sample_points.append([A[0],A[1]])
        return sample_points
    for k in np.arange(0.0, l2Dist, l2Dist/sample_dist):
        sample_point = [0,0]
        sample_point[0] = A[0] + (k/l2Dist)*vec_nonorm[0]
        sample_point[1] = A[1] + (k/l2Dist)*vec_nonorm[1]
        if prev_sp is not None:
            if getL2Dist(prev_sp, sample_point) < 5: continue 
        prev_sp = sample_point
        sample_points.append(sample_point)
    sample_points.append([B[0],B[1]])
    return sample_points

###########################


import sys
sys.path.insert(0, OP_CAFFE_TRAIN_PATH)
import opcaffe

def create_meta(points_3d, points_2d, img):
    metaData = opcaffe.MetaData()

    # 2D Joints
    opjoints = opcaffe.Joints()
    oppoints = []
    oppoints3D = []
    opviz = []
    for j in range(0, points_2d.shape[0]):
        point = opcaffe.Point2f(int(points_2d[j,0]),int(points_2d[j,1]))
        point3D = opcaffe.Point3f(points_3d[j,0], points_3d[j,1], points_3d[j,2])
        oppoints.append(point)
        oppoints3D.append(point3D)
        opviz.append(int(points_2d[j,2]))
    opjoints.points = oppoints
    opjoints.points3D = oppoints3D
    opjoints.isVisible = opviz
    metaData.jointsSelf = opjoints

    # Bbox
    rect = get_rect(points_2d)
    centroid = [rect[0] + (rect[2]-rect[0])/2, rect[1] + (rect[3]-rect[1])/2]
    metaData.objPos.x = centroid[0]
    metaData.objPos.y = centroid[1]

    # Size
    metaData.imageSize = opcaffe.Size(img.shape[1],img.shape[0])
    metaData.numberOtherPeople = 0
    metaData.scaleSelf = 1

    return metaData

def viz_hm(img, hm):
    for j in range(0, 25):
        image = img[0,:,:]+0.5
        #image = (image*255).astype(np.uint8)
        image_orig = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

        image = image_orig.copy()

        index = j

        hmrs = cv2.resize(hm[index,:,:].copy(), (368, 368), 0, 0, interpolation = cv2.INTER_CUBIC)

        image[:,:,0] += (np.abs(hmrs))
        image[:,:,1] += (np.abs(hmrs))
        image[:,:,2] += (np.abs(hmrs))

        cv2.imshow("HM", image)        

        cv2.waitKey(0)

def viz_pof(img, pof, paf):
    full_img = cv2.cvtColor(img[0,:,:]+0.5,cv2.COLOR_GRAY2BGR)

    for j in range(0, 24):
        image = img[0,:,:]+0.5
        #image = (image*255).astype(np.uint8)
        image_orig = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

        image = image_orig.copy()

        index = j

        pof_x = cv2.resize(pof[3*index + 0,:,:].copy(), (368, 368), 0, 0, interpolation = cv2.INTER_CUBIC)
        pof_y = cv2.resize(pof[3*index + 1,:,:].copy(), (368, 368), 0, 0, interpolation = cv2.INTER_CUBIC)
        pof_z = cv2.resize(pof[3*index + 2,:,:].copy(), (368, 368), 0, 0, interpolation = cv2.INTER_CUBIC)

        if paf is not None:
            paf_x = cv2.resize(paf[2*index + 0,:,:].copy(), (368, 368), 0, 0, interpolation = cv2.INTER_CUBIC)
            paf_y = cv2.resize(paf[2*index + 1,:,:].copy(), (368, 368), 0, 0, interpolation = cv2.INTER_CUBIC)

        scalar = 10
        for v in range(0, image.shape[0], 10):
            for u in range(0, image.shape[1], 10):
                if not pof_x[v,u] and not pof_y[v,u]: continue
                p1 = (u, v)
                p2 = (u + scalar*pof_x[v,u], v + scalar*pof_y[v,u])
                cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,255,0), 1)
                cv2.line(full_img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,255,0), 1)

        image[:,:,0] += (np.abs(pof_x))
        image[:,:,1] += (np.abs(pof_y))
        image[:,:,2] += (np.abs(pof_z))

        full_img[:,:,0] += (np.abs(pof_x))
        full_img[:,:,1] += (np.abs(pof_y))
        full_img[:,:,2] += (np.abs(pof_z))

        cv2.imshow("POF", image)

        if paf is not None:
            image = image_orig.copy()

            scalar = 10
            for v in range(0, image.shape[0], 10):
                for u in range(0, image.shape[1], 10):
                    if not paf_x[v,u] and not paf_y[v,u]: continue
                    p1 = (u, v)
                    p2 = (u + scalar*paf_x[v,u], v + scalar*paf_y[v,u])
                    cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,255,0), 1)

            image[:,:,0] += (np.abs(paf_x))
            image[:,:,1] += (np.abs(paf_y))

            cv2.imshow("PAF", image)        

        cv2.waitKey(0)

    cv2.imshow("full", full_img)
    cv2.waitKey(0)

class POFBodyLoader():
    def __init__(self, db_filename, batch_size, resolution=368):
        self.batch_size = batch_size
        params = {
            "batch_size" : 10,
            "stride": 8,
            "max_degree_rotations": "0.0", # !! WE DONT HANDLE POF ROTATIONS
            "crop_size_x": resolution,
            "crop_size_y": resolution,
            "center_perterb_max": 40.0,
            "center_swap_prob": 0.0,
            "scale_prob": 1.0,
            "scale_mins": "0.333333333333",
            "scale_maxs": "1.5",
            "target_dist": 0.600000023842,
            "number_max_occlusions": "2",
            "sigmas": "7.0",
            "model": "COCO_25B_17",
            "source_background": LMDB_BACKGROUND,
        }
        self.opTransformer = opcaffe.OPTransformer(params)

        self.db_data = pickle.load(open(db_filename, 'rb'))

        self.state = 0

    def get_index(self, i):
        img = cv2.imread(self.db_data["img_dirs"][i])
        K = self.db_data["K"][i]
        R = self.db_data["R"][i]
        t = self.db_data["t"][i]
        distCoef = self.db_data["distCoef"][i]
        calib = {"K": K, "R": R, "t": t, "distCoef": distCoef}
        joints = self.db_data["body"][i]

        # Convert to COCO Format
        joints = convert(joints) 

        # Project
        pt = project2D(joints, calib, imgwh=(img.shape[1], img.shape[0]), applyDistort=True)

        # Create elems
        points_3d = np.zeros((17,3))
        points_2d = np.zeros((17,3))
        for j in range(0, joints.shape[0]):
            point_2d = pt[0][:,j]
            valid = pt[1][j]
            point_3d = pt[2][:,j]
            if valid: 
                points_3d[j,:] = point_3d
                points_2d[j,:] = np.array([point_2d[0], point_2d[1], 1])
            else:
                points_3d[j,:] = np.array([0,0,0])
                points_2d[j,:] = np.array([0,0,2])

        # Stuff
        batch = opcaffe.Batch()
        metaData = create_meta(points_3d, points_2d, img)
        self.opTransformer.load(img, metaData, batch)
        #viz_pof(batch)

        # Return
        image = batch.data.copy()
        paf_mask = batch.label[:, 0:72, :, :].copy()
        hm_mask = batch.label[:, 72:97, :, :].copy()
        paf = batch.label[:, 97:169, :, :].copy()
        hm = batch.label[:, 169:194, :, :].copy()
        pof_mask = np.zeros((1,24*3,46,46), dtype=np.float32)
        counter = 0
        for j in range(0, 24*2, 2):
            pof_mask[:,counter*3 + 0, :, :] = paf_mask[:, j, :, :]
            pof_mask[:,counter*3 + 1, :, :] = paf_mask[:, j, :, :]
            pof_mask[:,counter*3 + 2, :, :] = paf_mask[:, j, :, :]
            counter+=1
        pof = batch.other.copy()

        return image, paf_mask, paf, pof_mask, pof, hm_mask, hm

    def get(self):

        # Get N
        N = len(self.db_data["body"])

        # Sample Batch size from X_train
        indexes = np.random.choice(N, self.batch_size)

        # print("HACCCK!")
        # indexes[0] = 93

        # Add back
        images = []
        paf_masks = []
        pafs = []
        pof_masks = []
        pofs = []
        hm_masks = []
        hms = []
        for i in range(0, indexes.shape[0]):
            image, paf_mask, paf, pof_mask, pof, hm_mask, hm = self.get_index(indexes[i])
            images.append(image)
            paf_masks.append(paf_mask)
            pafs.append(paf)
            pof_masks.append(pof_mask)
            pofs.append(pof)
            hm_masks.append(hm_mask)
            hms.append(hm)
        images = np.concatenate(images, axis=0)
        paf_masks = np.concatenate(paf_masks, axis=0)
        pafs = np.concatenate(pafs, axis=0)
        pof_masks = np.concatenate(pof_masks, axis=0)
        pofs = np.concatenate(pofs, axis=0)
        hm_masks = np.concatenate(hm_masks, axis=0)
        hms = np.concatenate(hms, axis=0)



        return images, paf_masks, pafs, pof_masks, pofs, hm_masks, hms

# PAF_type = 0
# allPAFConnection = [[np.array([[1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4], [2, 16], [1, 5], [5, 6], [6, 7], [5, 17], [1, 0], [0, 14], [0, 15], [14, 16], [15, 17], [1, 18], [1, 19], [19, 8], [19, 11]]),
#                      np.array([[0, 4], [4, 3], [3, 2], [2, 1], [0, 8], [8, 7], [7, 6], [6, 5], [0, 12], [12, 11], [11, 10], [10, 9], [0, 16], [16, 15], [15, 14], [14, 13], [0, 20], [20, 19], [19, 18], [18, 17]])
#                      ],  # PAF type 0 (Original Openpose)
#                     [np.array([[1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4], [2, 16], [1, 5], [5, 6], [6, 7], [5, 17],
#                                [1, 0], [0, 14], [0, 15], [14, 16], [15, 17], [1, 18], [2, 4], [5, 7], [8, 4], [11, 7], [8, 10], [11, 13]]),  # augmented PAF
#                      np.array([[0, 4], [4, 3], [3, 2], [2, 1], [0, 8], [8, 7], [7, 6], [6, 5], [0, 12], [12, 11], [11, 10], [10, 9], [0, 16], [16, 15], [15, 14], [14, 13], [0, 20], [20, 19], [19, 18], [18, 17]])
#                      ]]  # PAF type 1 (My augmented PAF)
# PAFConnection = allPAFConnection[PAF_type]

pof_a = [0,0,1,2,   0,0,   5,6,   7, 8,    5, 6,   11,12,   13,14,   15,19,15,  16,22,16,    5, 5]
pof_b = [1,2,3,4,   5,6,   7,8,   9,10,   11,12,   13,14,   15,16,   19,20,21,  22,23,24,   17,18]

def PAF_to_3D(coord2d, PAF, stride, objtype=0):

    print(PAF.shape)

    # Resize POF
    PAF_resized = np.zeros((PAF.shape[0], PAF.shape[1]*stride, PAF.shape[2]*stride), dtype=PAF.dtype)
    for j in range(0, PAF.shape[0]):
        PAF_resized[j,:,:] = cv2.resize(PAF[j,:,:], (PAF_resized.shape[2], PAF_resized.shape[1]), 0, 0, interpolation = cv2.INTER_CUBIC)
    PAF = PAF_resized

    # if objtype == 0:
    #     depth_root_idx = 1  # put neck at 0-depth
    # else:
    #     assert objtype == 1
    #     depth_root_idx = 0
    # assert len(coord2d.shape) == 2 and coord2d.shape[1] == 2
    coord3d = np.zeros((coord2d.shape[0], 3), dtype=coord2d.dtype)

    depth_root_idx = pof_b[0]

    coord3d[:, :2] = coord2d[:, :2]
    coord3d[depth_root_idx, 2] = 0.0
    vec3d_array = []


    for j in range(0, len(pof_a)):
        conn = (pof_a[j], pof_b[j])

        A = coord2d[conn[0]][0:3]
        B = coord2d[conn[1]][0:3]
        u = np.linspace(0.0, 1.0, num=11)
        v = 1.0 - u

        if A[2] < 0.05 or B[2] < 0.05: continue 

        points = (np.outer(A, v) + np.outer(B, u)).astype(int)  # 2 * N

        vec3ds = PAF[3 * j:3 * j + 3, points[1], points[0]].T   # note order of y, x in index

        vec3d = np.mean(vec3ds, axis=0)
        vec3d[np.isnan(vec3d)] = 0.0  # numerical stability

        if (A == B).all():  # A and B actually coincides with each other, put the default bone length.
            print("UNHANDLED")
            stop

        else:
            # find the least square solution of Ax = b
            A = np.zeros([3, 2])
            A[2, 0] = -1.
            A[:, 1] = vec3d
            b = coord3d[conn[1]] - coord3d[conn[0]]  # by this time the z-value of target joint should be 0
            x, _, _, _ = nl.lstsq(A, b, rcond=-1)

            if x[1] < 0:  # the direction is reversed
                if vec3d[2] >= 0:
                    coord3d[conn[1], 2] = coord3d[conn[0], 2] + 0.5  # assume that this connection is vertical to the screen
                else:
                    coord3d[conn[1], 2] = coord3d[conn[0], 2] - 0.5
            else:
                coord3d[conn[1], 2] = x[0]
            if nl.norm(vec3d) < 0.1 or x[1] < 0:  # If there is almost no response, or the direction is reversed, put it zero so that Adam does not fit.
                vec3d[:] = 0

    return coord3d

import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
once = False
fig = matplotlib.pyplot.figure()
ax  = fig.add_subplot(111, projection = '3d')
ax.set_xlim(0,500)
ax.set_ylim(0,500)
ax.set_zlim(0,500)

def viz_coord(coord3d, coord2d, mode=0):
    # DRAW
    global once

    for j in range(0, len(pof_a)):
        conn = (pof_a[j], pof_b[j])

        A = coord2d[conn[0]][0:3]
        B = coord2d[conn[1]][0:3]
        if A[2] < 0.05 or B[2] < 0.05: continue 

        A = coord3d[conn[0]]
        B = coord3d[conn[1]]

        ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], color = 'b')

    ax.set_xlim(0,300)
    ax.set_ylim(0,300)
    ax.set_zlim(0,300)

    if mode == 1:
        matplotlib.pyplot.show()
    else:
        matplotlib.pyplot.pause(0.05)
        matplotlib.pyplot.draw()
        ax.clear()

import time
if __name__ == '__main__':

    # Generate Data
    DomeReader(mode='training', shuffle=True, objtype=0, crop_noise=True, full_only=False, debug=False, savename="human3d.pkl")
    DomeReader(mode='training', shuffle=True, objtype=0, crop_noise=True, full_only=False, debug=True, savename="human3d_test.pkl")
