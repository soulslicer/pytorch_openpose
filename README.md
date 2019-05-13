## Openpose Training

```
1. Clone OpenPose and set BUILD_PYTHON=ON (https://github.com/CMU-Perceptual-Computing-Lab/openpose)
2. Clone my openpose_caffe_train code (https://github.com/soulslicer/openpose_caffe_train/)
	cd openpose_caffe_train
	git checkout new_commits_fix
	git submodule init
	git submodule update --init --recursive
	mkdir build; cd build; cmake..; make -j10
3. Go to train_gines.py and set the paths correctly
	NAME = "weights_gines_no"
	OP_CAFFE_TRAIN_PATH = '/home/raaj/openpose_caffe_train/build/op/'
	OP_PYTHON_PATH = '/home/raaj/openpose_orig/build/python/'
	OP_MODEL_FOLDER = '/home/raaj/openpose_orig/models/'
	OP_LMDB_FOLDER = '/media/raaj/Storage/openpose_train/dataset/'
	OP_RESOLUTION = 480
4. To start Training
	python train_gines.py --batch 24 --ngpu 4 --debug 0
5. To visualize result from Openpose
	python train_gines.py --batch 1 --ngpu 1 --debug 1
6. To retrain from scratch
	python train_gines.py --batch 24 --ngpu 4 --debug 0 --reload 
7. To copy weights from caffe model, enable this line:
	model.net.load_caffe() 
8. To Evaluate
	sh get_validation_data.sh
	python test_gines.py --weight NAME/model.pth
	python coco_eval.py
```

## POF Body Training

```
1. Go to pof.py and change variables
	IMAGE_ROOT = '/media/posefs0c/panopticdb/'
	TOTALPOSE_ROOT = '/media/posefs0c/Users/donglaix/Experiments/totalPose/'
	OP_CAFFE_TRAIN_PATH = '/home/raaj/openpose_caffe_train/build/op/'
	LMDB_BACKGROUND = "/media/raaj/Storage/openpose_train/dataset/lmdb_background"
2. Run pof.py
	python pof.py
3. Train pof. Jointly trains POF Module and OpenPose together
	python train_pof.py
4. Test pof (Need Camera)
	python test_pof.py --weight WEIGHTHERE
```
