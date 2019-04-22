import numpy as np
import numpy.linalg as nl
import pickle
import json
import os

DATASET_LOCATION = "/media/posefs0c/panopticdb/a4/"

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


pof_a = [0,0,1,2,   0,0,   5,6,   7, 8,    5, 6,   11,12,   13,14,   15,19,15,  16,22,16,    5, 5]
pof_b = [1,2,3,4,   5,6,   7,8,   9,10,   11,12,   13,14,   15,16,   19,20,21,  22,23,24,   17,18]


def convert(points):
    new_points = np.zeros((25,3))



    for i in range(0, new_points.shape[0]):

        try:
            new_points[i, :] = points[dome_to_body25b[i], :]
        except:
            new_points[i, :] = np.array([50000,50000,0.0001])

        # if i in dome_to_body25b.keys():
        #     new_points[i, :] = points[dome_to_body25b[i], :]
        # else:
        #     new_points[i, :] = np.array([0,0,-1])
    return new_points

class DomeReader():

    def __init__(self, mode='training', objtype=0, shuffle=False, batch_size=1, crop_noise=False, full_only=True, head_top=True):
        self.image_root = '/media/posefs0c/panopticdb/'
        self.totalpose_root = '/media/posefs0c/Users/donglaix/Experiments/totalPose/'

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
            if counter == 5: break
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
            if counter == 5: break
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

        pickle.dump(human3d, open('human3d.pkl',"wb"), protocol=2)

        # import cv2
        # for img_dir in img_dirs:
        #     if cv2.imread(img_dir) is None:
        #         print(img_dir)

        # self.register_tensor(human3d, order_dict)
        # self.num_samples = len(self.tensor_dict['img_dirs'])

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

    if imgwh is not None:
        assert len(imgwh) == 2
        imw, imh = imgwh
        winside_img = np.logical_and(pt[0, :] > -0.5, pt[0, :] < imw-0.5) 
        hinside_img = np.logical_and(pt[1, :] > -0.5, pt[1, :] < imh-0.5) 
        inside_img = np.logical_and(winside_img, hinside_img) 
        inside_img = np.logical_and(inside_img, R2 < 1.0) 

        return pt, inside_img, x

    stop


if __name__ == '__main__':
    #d = DomeReader(mode='training', shuffle=True, objtype=0, crop_noise=True, full_only=False)

    db_data = pickle.load(open("human3d.pkl", 'rb'))


    print(len(db_data["body"])) # 244 3D Keypoints

    print(len(db_data["K"])) # 244 Timestamp calibreations

    print(len(db_data["img_dirs"])) # 244 Image Datas

    for i in range(0, len(db_data["body"])):
        img = cv2.imread(db_data["img_dirs"][i])
        K = db_data["K"][i]
        R = db_data["R"][i]
        t = db_data["t"][i]
        distCoef = db_data["distCoef"][i]
        calib = {"K": K, "R": R, "t": t, "distCoef": distCoef}
        joints = db_data["body"][i]

        # Convert to Body 25
        joints = convert(joints) 

        # Project
        pt = project2D(joints, calib, imgwh=(img.shape[1], img.shape[0]), applyDistort=True)

        # Create elems
        points_3d = np.zeros((25,3))
        points_2d = np.zeros((25,3))
        for j in range(0, joints.shape[0]):
            point_2d = pt[0][:,j]
            valid = pt[1][j]
            point_3d = pt[2][:,j]
            if valid: 
                points_3d[j,:] = point_3d
                points_2d[j,:] = np.array([point_2d[0], point_2d[1], 1])
            else:
                points_3d[j,:] = np.array([0,0,0])
                points_2d[j,:] = np.array([0,0,0])


        

        # Vis 2D
        for j in range(0, joints.shape[0]):
            if points_2d[j,2] == 0: continue
            point_2d = points_2d[j,:]
            cv2.putText(img,str(j), (int(point_2d[0]), int(point_2d[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            cv2.circle(img,(int(point_2d[0]), int(point_2d[1])), 5, (0,255,0), -1)

        # Create POF

        # # Vis POF
        # for j in range(0, len(pof_a)):
        #     p1 = points_2d[]


        cv2.imshow("win", img)
        cv2.waitKey(0)

        #stop

        # Ks = db_data["body"][i]
        # Ks = db_data["body"][i]
        # Ks = db_data["body"][i]



        #print(body.shape)


    # human3d = {'body': [], 'left_hand': [], 'right_hand': [], 'body_valid': [], 'left_hand_valid': [], 'right_hand_valid': []}
    # calib = {'K': [], 'R': [], 't': [], 'distCoef': []}
    # img_dirs = []


    # # d.rotate_augmentation = True
    # # d.blur_augmentation = True
    # data_dict = d.get()
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # sess.run(tf.global_variables_initializer())
    # tf.train.start_queue_runners(sess=sess)

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # import utils.general
    # from utils.vis_heatmap3d import vis_heatmap3d
    # from utils.PAF import plot_PAF, PAF_to_3D, plot_all_PAF

    # validation_images = []

    # for i in range(d.num_samples):
    #     print('{}/{}'.format(i + 1, d.num_samples))
    #     values = \
    #         sess.run([data_dict['image_crop'], data_dict['img_dir'], data_dict['keypoint_uv_local'], data_dict['hand_valid'], data_dict['scoremap2d'],
    #                   data_dict['PAF'], data_dict['mask_crop'], data_dict['keypoint_xyz_local']])
    #     image_crop, img_dir, hand2d, hand_valid, hand2d_heatmap, PAF, mask_crop, hand3d = [np.squeeze(_) for _ in values]

    #     image_name = img_dir.item().decode()
    #     image_v = ((image_crop + 0.5) * 255).astype(np.uint8)

    #     hand2d_detected, bscore = utils.PAF.detect_keypoints2d_PAF(hand2d_heatmap, PAF, objtype=1)
    #     # hand2d_detected = utils.general.detect_keypoints2d(hand2d_heatmap)[:20, :]
    #     hand3d_detected, _ = PAF_to_3D(hand2d_detected, PAF, objtype=1)
    #     hand3d_detected = hand3d_detected[:21, :]

    #     fig = plt.figure(1)
    #     ax1 = fig.add_subplot(231)
    #     plt.imshow(image_v)
    #     utils.general.plot2d(ax1, hand2d, type_str='hand', valid_idx=hand_valid, color=np.array([1.0, 0.0, 0.0]))
    #     utils.general.plot2d(ax1, hand2d_detected, type_str='hand', valid_idx=hand_valid, color=np.array([0.0, 0.0, 1.0]))

    #     ax2 = fig.add_subplot(232, projection='3d')
    #     utils.general.plot3d(ax2, hand3d_detected, type_str='hand', valid_idx=hand_valid, color=np.array([0.0, 0.0, 1.0]))
    #     ax2.set_xlabel('X Label')
    #     ax2.set_ylabel('Y Label')
    #     ax2.set_zlabel('Z Label')
    #     plt.axis('equal')

    #     ax3 = fig.add_subplot(233, projection='3d')
    #     utils.general.plot3d(ax3, hand3d, type_str='hand', valid_idx=hand_valid, color=np.array([1.0, 0.0, 0.0]))
    #     ax2.set_xlabel('X Label')
    #     ax2.set_ylabel('Y Label')
    #     ax2.set_zlabel('Z Label')
    #     plt.axis('equal')

    #     xy, z = plot_all_PAF(PAF, 3)
    #     ax4 = fig.add_subplot(234)
    #     ax4.imshow(xy)

    #     ax5 = fig.add_subplot(235)
    #     ax5.imshow(z)

    #     plt.show()

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

# VGG is unlocked ?
# WHen OPLoader, Lock POF, Unlock PAF,KP
# When Dome, Lock PAF, KP, Unlock POF