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

MAX_REF = 1175

def decision(probability):
    return random.random() < probability

def get_rand_int(start, end, exc=-1):
    #random.seed(time.clock()) ### Disable this if you want same value
    while True:
        y = random.randint(start, end)
        if y == exc: continue
        return y

# filenames = {'train': ['class_tripletlist_train.txt', 'closure_tripletlist_train.txt', 
#                 'gender_tripletlist_train.txt', 'heel_tripletlist_train.txt'],
#              'val': ['class_tripletlist_val.txt', 'closure_tripletlist_val.txt', 
#                 'gender_tripletlist_val.txt', 'heel_tripletlist_val.txt'],
#              'test': ['class_tripletlist_test.txt', 'closure_tripletlist_test.txt', 
#                 'gender_tripletlist_test.txt', 'heel_tripletlist_test.txt']}

def default_image_loader(path):
    return Image.open(path).convert('L')

def get_padded(val):
    return '{0:05d}'.format(val)

def load_csv(file, range_wanted):
    f = open(file, 'r')
    x = f.readlines()
    arrs = []
    for string in x:
        string = string.rstrip()
        arr = string.split(",")
        arr = [int(arr[0]), int(arr[1]), arr[2]]
        if arr[0] < range_wanted[0] or arr[0] > range_wanted[1]: continue 
        arrs.append(arr)
    f.close()
    return arrs

def show_image(img, wait=2,name="win2"):
    if img is not None:
        cv2.imshow(name,img)
        key = cv2.waitKey(wait)
        if key == 32:
            while 1:
                key = cv2.waitKey(2)
                if key == 32:
                    break
        return key

def pil_from_np(img):
    return Image.fromarray(img, 'L')

def np_from_pil(img):
    return np.array(img)[0,:,:]

def apply_transform_same(transform, img1, img2):
    seed = np.random.randint(2147483647) # make a seed with numpy generator 
    random.seed(seed) # apply this seed to img tranfsorms
    img1 = transform(img1)
            
    random.seed(seed) # apply this seed to target tranfsorms
    img2 = transform(img2)

    return img1, img2

def rotate_and_scale(image, angle, scale, trans_x, trans_y):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[0:2, 0:2]*=scale
    rot_mat[0,2] += trans_x
    rot_mat[1,2] += trans_y
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(128))
    return result

def add_gray(image, si, di):
    c1 = int(image.shape[0]*si)
    c2 = int(image.shape[0]*di)
    image[0:c1, :] = 128
    image[image.shape[0]-c2:image.shape[0]] = 128
    return image

def show_images(img_arr, img_arr2, name="data"):
    if img_arr is not None:
        img1 = img_arr[0]
        img2 = img_arr[1]
        img3 = img_arr[2]

        # Visualize
        vis = np.concatenate((np_from_pil(img1), np_from_pil(img2), np_from_pil(img3)), axis=1)
        #cv2.putText(vis, 0, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(name,vis)
    if img_arr2 is not None:
        img1 = img_arr2[0]
        img2 = img_arr2[1]
        img3 = img_arr2[2]

        # Visualize
        vis = np.concatenate((np_from_pil(img1), np_from_pil(img2), np_from_pil(img3)), axis=1)
        #cv2.putText(vis, 0, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(name+"2",vis)
    
    cv2.waitKey(-1)



class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, range_wanted, transform=None, loader=default_image_loader):
        """ filenames_filename: A text file with each line containing the path to an image e.g.,
                images/class1/sample.jpg
            triplets_file_name: A text file with each line containing three integers, 
                where integer i refers to the i-th image in the filenames file. 
                For a line of intergers 'a b c', a triplet is defined such that image a is more 
                similar to image c than it is to image b, e.g., 
                0 2017 42 """
        self.root = root
        data = load_csv(self.root + 'label_table.csv', range_wanted)

        self.transform = transform
        self.loader = loader
        self.triplets = data

        self.counter = 0

        # self.angle_mod = 10
        # self.scale_mod = 0.1
        # self.trans_mod = 10

        self.angle_mod = 5
        self.scale_mod = 0.05
        self.trans_mod = 5

        self.real_mode_prob = 0

        self.cut_effect_prob = 0

        self.debug_viz = True

    def generate_noise_image(self, img, amt = 5):
        # Get a noise profile
        while 1:
            index = get_rand_int(0, len(self.triplets)-1)
            triplet = self.triplets[index]      
            if triplet[2] == "full": break

        # Load image
        anchor_index = triplet[0]
        pos_index = triplet[1]
        anchor_image = cv2.imread(self.root + "tracks_cropped/" + get_padded(anchor_index) + ".jpg", 0)
        pos_image = cv2.imread(self.root + "references/" + get_padded(pos_index) + ".png", 0)
        pos_image = cv2.resize(pos_image, (anchor_image.shape[1],anchor_image.shape[0])) 
        pos_image = pos_image.astype(np.float32)/255.
        anchor_image = anchor_image.astype(np.float32)/255.
        noise_profile = np.abs(pos_image - anchor_image)
        noise_profile = anchor_image
        #noise_profile = cv2.flip(noise_profile, -1)

        # Resize clean image
        clean = img.copy()
        clean = clean.astype(np.float32)/255.
        noise_profile_res = cv2.resize(noise_profile, (clean.shape[1],clean.shape[0])) 
        clean_noise = clean

        # Add noise
        for i in range(0, amt):
            clean_noise = skimage.util.random_noise(clean_noise, mode='gaussian', seed=None, clip=True)
            clean_noise = skimage.util.random_noise(clean_noise, mode='Poisson', seed=None, clip=True)
            translation_matrix = np.float32([ [1,0,get_rand_int(-2,2,100)], [0,1,get_rand_int(-2,2,100)] ])
            noise_profile_res = cv2.warpAffine(noise_profile_res, translation_matrix, (noise_profile_res.shape[1], noise_profile_res.shape[0]))
            translation_matrix = np.float32([ [1,0,get_rand_int(-2,2,100)], [0,1,get_rand_int(-2,2,100)] ])
            clean_noise = cv2.warpAffine(clean_noise, translation_matrix, (clean_noise.shape[1], clean_noise.shape[0]))
            translation_matrix = np.float32([ [get_rand_int(1,2,100),0,get_rand_int(-2,2,100)], [0,get_rand_int(1,2,100),get_rand_int(-2,2,100)] ])
            weird_noise = cv2.warpAffine(clean_noise, translation_matrix, (clean_noise.shape[1], clean_noise.shape[0]))
            val = 0.1
            clean_noise = noise_profile_res*val+ clean_noise*(1-val) + (weird_noise-1)*0.1

        # Return
        clean_noise = (clean_noise*255).astype(np.uint8)
        return clean_noise

    def __getitem__(self, index):

        # Hack
        index = self.counter
        self.counter += 1

        # Mode Fake mode vs Real mode
        real_mode = decision(self.real_mode_prob)

        # Real Mode
        if real_mode:

            # Get Anchor and Pos Index
            anchor_index = self.triplets[index][0]
            pos_index = self.triplets[index][1]

            # Get Neg Index - so that it is not same as pos_index
            neg_index = get_rand_int(1, MAX_REF, pos_index)

            # Load files
            anchor_image = cv2.imread(self.root + "tracks_cropped/" + get_padded(anchor_index) + ".jpg", 0)
            pos_image = cv2.imread(self.root + "references/" + get_padded(pos_index) + ".png", 0)
            neg_image = cv2.imread(self.root + "references/" + get_padded(neg_index) + ".png", 0)

            # Anchor crop
            anchor_crop_mode = self.triplets[index][2]

            # Not sure how to handle
            if anchor_crop_mode == "-":
                anchor_crop_mode = "middle"

            # Add Borders and chop borders
            if anchor_crop_mode == "top":
                anchor_image = np.vstack((anchor_image,np.ones(shape=((anchor_image.shape[0]),anchor_image.shape[1]),dtype=np.uint8)*128))
                pos_image[pos_image.shape[0]/2:pos_image.shape[0],:] = 128
            elif anchor_crop_mode == "middle":
                og = anchor_image.shape[0]
                anchor_image = np.vstack((anchor_image,np.ones(shape=((og)/2,anchor_image.shape[1]),dtype=np.uint8)*128))
                anchor_image = np.vstack((np.ones(shape=((og)/2,anchor_image.shape[1]),dtype=np.uint8)*128, anchor_image))
                pos_image[0:pos_image.shape[0]/4,:] = 128
                pos_image[(pos_image.shape[0]/4)*3:pos_image.shape[0]:] = 128        
            if anchor_crop_mode == "bottom":
                anchor_image = np.vstack((np.ones(shape=((anchor_image.shape[0]),anchor_image.shape[1]),dtype=np.uint8)*128,anchor_image))
                pos_image[0:pos_image.shape[0]/2,:] = 128

            # Flip Image
            if decision(0.5):
                anchor_image = cv2.flip(anchor_image, 1 )
                pos_image = cv2.flip(pos_image, 1 )
            if decision(0.5):
                neg_image = cv2.flip(neg_image, 1 )

            # Input Images have Aug
            anchor_image_inp = anchor_image.copy()
            pos_image_inp = pos_image.copy()
            neg_image_inp = neg_image.copy()
            anchor_image_inp = rotate_and_scale(anchor_image_inp, angle=get_rand_int(-self.angle_mod,self.angle_mod), scale=random.uniform(1-self.scale_mod, 1+self.scale_mod), trans_x=get_rand_int(-self.trans_mod,self.trans_mod), trans_y=get_rand_int(-self.trans_mod,self.trans_mod))
            pos_image_inp = rotate_and_scale(pos_image_inp, angle=get_rand_int(-self.angle_mod,self.angle_mod), scale=random.uniform(1-self.scale_mod, 1+self.scale_mod), trans_x=get_rand_int(-self.trans_mod,self.trans_mod), trans_y=get_rand_int(-self.trans_mod,self.trans_mod))
            neg_image_inp = rotate_and_scale(neg_image_inp, angle=get_rand_int(-self.angle_mod,self.angle_mod), scale=random.uniform(1-self.scale_mod, 1+self.scale_mod), trans_x=get_rand_int(-self.trans_mod,self.trans_mod), trans_y=get_rand_int(-self.trans_mod,self.trans_mod))

        # Fake Mode
        else:

            # # Get Anchor and Pos Index
            anchor_index = get_rand_int(1, MAX_REF)
            pos_index = anchor_index

            #print anchor_index

            # Get Neg Index - so that it is not same as pos_index
            neg_index = get_rand_int(1, MAX_REF, pos_index)

            # Load images
            anchor_image = cv2.imread(self.root + "references/" + get_padded(anchor_index) + ".png", 0)
            pos_image = anchor_image.copy()
            neg_image = cv2.imread(self.root + "references/" + get_padded(neg_index) + ".png", 0)

            # Add fake noise
            anchor_image = self.generate_noise_image(anchor_image, get_rand_int(1,5))

            # Add Cut effect
            if decision(self.cut_effect_prob):
                top_cut = random.uniform(0, 0.5)
                bottom_cut = random.uniform(0, 0.5-top_cut)
                anchor_image = add_gray(anchor_image, top_cut, bottom_cut)
                pos_image = add_gray(pos_image, top_cut, bottom_cut)

            # Flip Image
            if decision(0.5):
                anchor_image = cv2.flip(anchor_image, 1)
                pos_image = cv2.flip(pos_image, 1 )
            if decision(0.5):
                neg_image = cv2.flip(neg_image, 1 )

            # Input Images have Aug
            anchor_image_inp = anchor_image.copy()
            pos_image_inp = pos_image.copy()
            neg_image_inp = neg_image.copy()
            anchor_image_inp = rotate_and_scale(anchor_image_inp, angle=get_rand_int(-self.angle_mod,self.angle_mod), scale=random.uniform(1-self.scale_mod, 1+self.scale_mod), trans_x=get_rand_int(-self.trans_mod,self.trans_mod), trans_y=get_rand_int(-self.trans_mod,self.trans_mod))
            pos_image_inp = rotate_and_scale(pos_image_inp, angle=get_rand_int(-self.angle_mod,self.angle_mod), scale=random.uniform(1-self.scale_mod, 1+self.scale_mod), trans_x=get_rand_int(-self.trans_mod,self.trans_mod), trans_y=get_rand_int(-self.trans_mod,self.trans_mod))
            neg_image_inp = rotate_and_scale(neg_image_inp, angle=get_rand_int(-self.angle_mod,self.angle_mod), scale=random.uniform(1-self.scale_mod, 1+self.scale_mod), trans_x=get_rand_int(-self.trans_mod,self.trans_mod), trans_y=get_rand_int(-self.trans_mod,self.trans_mod))


        # Convert 
        img1, img2 = apply_transform_same(self.transform, pil_from_np(anchor_image), pil_from_np(pos_image))
        img3 = self.transform(pil_from_np(neg_image))
        img1_inp, img2_inp = apply_transform_same(self.transform, pil_from_np(anchor_image_inp), pil_from_np(pos_image_inp))
        img3_inp = self.transform(pil_from_np(neg_image_inp))

        # Visualize
        if self.debug_viz: show_images([img1, img2, img3], [img1_inp, img2_inp, img3_inp])

        return img1, img2, img3, img1_inp, img2_inp, img3_inp


        # print index
        # stop

        # return None, None, None
        
        # stop

        #path1, path2, path3, c = self.triplets[index]

        #print path1
        # During this point, it gets an index on self.triplets and runs
        # We could store data as only anchor and positive

        # Training
        # 2 Sets of data
        # Real - Anchor and positive set as IDs
        # Fake - Just a list of the same as IDs
        # During Sampling
            # For Real - For Negative, random sample from either fake or real (add more noise here?)
            # For fake - For Positive, use back same but add noise. For Negative, same as above 

        # Mechanism for blocking bounding box - top, middle, bottom

        if os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)])) and os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)])) and os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)])):
            img1 = self.loader(os.path.join(self.root, self.base_path, self.filenamelist[int(path1)]))
            img2 = self.loader(os.path.join(self.root, self.base_path, self.filenamelist[int(path2)]))
            img3 = self.loader(os.path.join(self.root, self.base_path, self.filenamelist[int(path3)]))
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                img3 = self.transform(img3)

            print img1.shape

            return img1, img2, img3, c
        else:
            return None

    def __len__(self):
        return len(self.triplets)
