sGenerateLightVersionForDebugging = True
# sGenerateLightVersionForDebugging = True
sDatasetFolder = "../dataset/hand/"
sDisplayDebugging = sGenerateLightVersionForDebugging

import cv2
import glob
import natsort
import json
import numpy as np
from coco_util import *

dpath = "/media/posefs0c/panopticdb/a4/"
json_data = load_json(dpath + "sample_list_2d.json")

print "0 - Not annotated, out of image or poor confidence"
print "1 - Annotated and not visibile"
print "2 - Annotated and visibile"

def load_calib_data(calib_file):
    with open(calib_file) as f:
        calib = json.load(f)
        for key, val in calib.iteritems():
            if type(val) == list:
                calib[key] = np.array(val)
        return calib
    stop

def vflag_to_color(vflag):
    if vflag == 0:
        return (0,0,0)
    elif vflag == 1:
        return (0,0,255)
    elif vflag == 2:
        return (0,255,0)

def runHands():
    # image_files = natsort.natsorted(glob.glob(imageAndAnnotFolder + "*.jpg"))
    # pt_files = natsort.natsorted(glob.glob(imageAndAnnotFolder + "*.json"))
    # print("Initial #annotations: " + str(len(image_files)))

    # # Consolidate data
    # hands_dict = dict()
    # for image_file, pt_file in zip(image_files, pt_files):
    #     img_id, pid_ifile, hand_dir = split_file(image_file)
    #     pt_id, pid_ptfile, pt_dir = split_file(pt_file)
    #     if (img_id[1] != pt_id[1]) or (pid_ifile[1] != pid_ptfile[1]):
    #         print("Error")
    #         stop
    #     if img_id not in hands_dict:
    #         hands_dict[img_id] = dict()
    #         hands_dict[img_id]["persons"] = dict()
    #         hands_dict[img_id]["image_path"] = image_file
    #     if pid_ifile not in hands_dict[img_id]["persons"]:
    #         hands_dict[img_id]["persons"][pid_ifile] = dict()
    #     if hand_dir not in hands_dict[img_id]["persons"][pid_ifile]:
    #         hands_dict[img_id]["persons"][pid_ifile][hand_dir] = dict()
    #     hands_dict[img_id]["persons"][pid_ifile][hand_dir]["pt_path"] = pt_file

    # totalWriteCount = len(hands_dict)
    # printEveryXIterations = max(1, round(totalWriteCount / 10))
    # print("Real #images: " + str(totalWriteCount))

    camera_cache = dict()

    images = []
    annotations = []
    ii = -1
    for data in json_data:
        ii += 1
        if sGenerateLightVersionForDebugging:
            if ii > 100: break
        #if ii == 0: continue

        # Body - If Visible - Occluded flag shows all correct
        #        Not visible - It shows all

        # Image
        image_path = data["img"]
        img = None
        print image_path
        img = cv2.imread(dpath + image_path)
        img_debug = None
        if sGenerateLightVersionForDebugging:
            img_debug = img.copy()
        img_width = img.shape[1]
        img_height = img.shape[0]

        # Image Object
        image_object = dict()
        image_object["id"] = ii
        image_object["file_name"] = image_path.split("/")[-1]
        image_object["width"] = img_width
        image_object["height"] = img_height
        images.append(image_object)

        # Valid
        subjects_with_valid_body = data["subjectsWithValidBody"]
        subjects_with_valid_Lhand = data["subjectsWithValidLHand"]
        subjects_with_valid_Rhand = data["subjectsWithValidRHand"] 

        # Body invalid but hand valid?

        seq_name = data["img"].split("/")[1]
        frame_name = data["img"].split("/")[2]
        camera_id = data["img"].split("/")[3].split("_")[0] + "_" + data["img"].split("/")[3].split("_")[1]
        camera_id_x = int(data["img"].split("/")[3].split("_")[1])

        # Load camera param
        if seq_name not in camera_cache.keys():
            camera_cache[seq_name] = dict()
        if camera_id not in camera_cache[seq_name]:
            camera_cache[seq_name][camera_id] = dict()
        if not len(camera_cache[seq_name][camera_id].keys()):
            camera_cache[seq_name][camera_id] = load_calib_data(dpath + "/annot_calib/" + seq_name + "/" + "calib_" + camera_id + ".json")
        calib = camera_cache[seq_name][camera_id]

        # CHeck the flag subjectsIthValidLHand check it! # Trust this only
        # Onccluded - Occluded by someone else - Never triggered if 1 person
        # Self-Occluded - Itself

        annot2d = load_json(dpath + data["annot_2d"])
        
        # Load Face Data
        face_data_3d = load_json("/media/posefs0c/panoptic/" + seq_name + "/hdFace3d/" + "faceRecon3D_hd"+frame_name+".json")
        face_data_2d = dict()
        for face_data in face_data_3d["people"]:
            face_id = face_data["id"]
            face_kp = np.array(face_data["face70"]["landmarks"])
            face_kp = face_kp.reshape((70,3))
            face_kp_2d = project2D(face_kp, calib, imgwh=(img_width, img_height), applyDistort=True)

            # Add based on Rules
            proj_data = []
            for i in range(0, 70):
                if not face_kp_2d[1][i]:
                    proj_data.append([0,0,0])
                    continue
                if face_data["face70"]["averageScore"][i] < 0.05:
                    proj_data.append([0,0,0])
                    continue

                if camera_id_x not in face_data["face70"]["visibility"][i]:
                    proj_data.append([face_kp_2d[0][0,i],face_kp_2d[0][1,i],1])
                    continue

                proj_point = (face_kp_2d[0][0,i],face_kp_2d[0][1,i],2)
                proj_data.append(proj_point)

            face_data_2d[face_id] = proj_data

        # Anno object
        person_array = []

        # Iterate each person
        for person_data in annot2d:
            pid = person_data["id"]

            # Body KPS
            body_kps = []
            if pid in subjects_with_valid_body:
                for i in range(0, 19):
                    confidence = person_data["body"]["scores"][i]
                    inside_img = person_data["body"]["insideImg"][i]
                    occluded = person_data["body"]["occluded"][i]
                    vflag = 2
                    if occluded: vflag = 1
                    if confidence < 0.05 or inside_img == 0:
                        body_kps.append([0,0,0])
                        continue   
                    body_kp = (person_data["body"]["landmarks"][i][0], 
                               person_data["body"]["landmarks"][i][1], 
                               vflag)
                    body_kps.append(body_kp)
                    if img_debug is not None: 
                        cv2.circle(img_debug,(int(body_kp[0]), int(body_kp[1])), 3, vflag_to_color(body_kp[2]), -1)
            else:
                stop

            # LHand KPS
            lhand_kps = []
            if pid in subjects_with_valid_Lhand:
                for i in range(0, 21):
                    confidence = person_data["left_hand"]["scores"][i]
                    inside_img = person_data["left_hand"]["insideImg"][i]
                    occluded = person_data["left_hand"]["self_occluded"][i]
                    vflag = 2
                    if occluded: vflag = 1
                    if confidence < 0.05 or inside_img == 0:
                        lhand_kps.append([0,0,0])
                        continue   
                    hand_kp = (person_data["left_hand"]["landmarks"][i][0], 
                               person_data["left_hand"]["landmarks"][i][1], 
                               vflag)
                    lhand_kps.append(hand_kp)
                    if img_debug is not None: 
                        cv2.circle(img_debug,(int(hand_kp[0]), int(hand_kp[1])), 3, vflag_to_color(hand_kp[2]), -1)
            else:
                # Need to create a bbox mask
                points = []
                for i in range(0, 21):  
                    kp = (person_data["left_hand"]["landmarks"][i][0], 
                               person_data["left_hand"]["landmarks"][i][1])
                    points.append(kp)
                rect = get_rect_from_points_only_bigger(points, img_width, img_height)
                if img_debug is not None: cv2.rectangle(img_debug, (rect[0], rect[1]), (rect[2], rect[3]), (255,0,0), 2)

                for i in range(0, 21):
                    lhand_kps.append([0,0,0])

            # LHand KPS
            rhand_kps = []
            if pid in subjects_with_valid_Rhand:
                for i in range(0, 21):
                    confidence = person_data["right_hand"]["scores"][i]
                    inside_img = person_data["right_hand"]["insideImg"][i]
                    occluded = person_data["right_hand"]["self_occluded"][i]
                    vflag = 2
                    if occluded: vflag = 1
                    if confidence < 0.05 or inside_img == 0:
                        rhand_kps.append([0,0,0]) 
                        continue   
                    hand_kp = (person_data["right_hand"]["landmarks"][i][0], 
                               person_data["right_hand"]["landmarks"][i][1], 
                               vflag)
                    rhand_kps.append(hand_kp)
                    if img_debug is not None: 
                        cv2.circle(img_debug,(int(hand_kp[0]), int(hand_kp[1])), 3, vflag_to_color(hand_kp[2]), -1)
            else:
                # Need to create a bbox mask
                points = []
                for i in range(0, 21):  
                    kp = (person_data["right_hand"]["landmarks"][i][0], 
                               person_data["right_hand"]["landmarks"][i][1])
                    points.append(kp)
                rect = get_rect_from_points_only_bigger(points, img_width, img_height)
                if img_debug is not None: cv2.rectangle(img_debug, (rect[0], rect[1]), (rect[2], rect[3]), (255,0,0), 2)

                for i in range(0, 21):
                    rhand_kps.append([0,0,0])  

            # If KP missing for hand, take the points and mask out  

            # Face
            face_kps = []
            if pid in face_data_2d.keys():
                fkps = face_data_2d[pid]
                for kp in fkps:
                    face_kps.append(kp)
                for kp in fkps:
                    if img_debug is not None:
                        cv2.circle(img_debug,(int(kp[0]), int(kp[1])), 3, vflag_to_color(kp[2]), -1)
            else:
                for i in range(0, 70):
                    face_kps.append([0,0,0])

            # 19 + 21 + 21 +70
            all_kps = body_kps + lhand_kps + rhand_kps + face_kps

            # Get rectangle
            rect = get_rect_from_points_only_bigger(all_kps, img_width, img_height, 0.1)
            rectW = rect[2]-rect[0]
            rectH = rect[3]-rect[1]
            # Display - Rectangle
            if img_debug is not None:
                cv2.rectangle(img_debug, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), 255, 2)

            # Store Person Data
            data = dict()
            data["segmentation"] = [] # DONT HAVE
            data["num_keypoints"] = len(all_kps)/3
            data["img_path"] = image_path.split("/")[-1]
            data["bbox"] = [rect[0], rect[1], rectW, rectH]
            data["area"] = data["bbox"][2]*data["bbox"][3]
            data["iscrowd"] = 0
            data["keypoints"] = np.array(all_kps).ravel().tolist()
            data["img_width"] = img_width
            data["img_height"] = img_height
            data["category_id"] = 1
            data["image_id"] = ii
            data["id"] = pid

            person_array.append(data)

        # Append Annot
        for arr in person_array:
            annotations.append(arr)

        # Visualize
        if img_debug is not None:
            cv2.imshow("win", img_debug)
            cv2.waitKey(0)

        # Create 100 images thats all

    # Json Object
    json_object = dict()
    json_object["info"] = dict()
    json_object["info"]["version"] = 1.0
    json_object["info"]["description"] = "Hands MPII Dataset in COCO Json Format"
    json_object["licenses"] = []
    json_object["images"] = images
    json_object["annotations"] = annotations

    # JSON writing
    jsonOutput = "dome_test.json"
    print("Saving " + jsonOutput + "...")
    print("Final #Images: " + str(len(json_object["images"])))
    print("Final #Annotations: " + str(len(json_object["annotations"])))
    open(jsonOutput, 'w').close()
    with open(jsonOutput, 'w') as outfile:
        json.dump(json_object, outfile)
    print("Saved!")


# # Test
# sImageAndAnnotFolder = sDatasetFolder + "hand_labels/manual_test/"
# sJsonOutput = sDatasetFolder + 'json/hand42_mpii_test.json'
# runHands(sJsonOutput, sImageAndAnnotFolder)
# print(' ')
# # Train
# sImageAndAnnotFolder = sDatasetFolder + "hand_labels/manual_train/"
# sJsonOutput = sDatasetFolder + 'json/hand42_mpii_train.json'
# runHands(sJsonOutput, sImageAndAnnotFolder)

runHands()