import cv2
import os,pandas
import shutil



def image_resizer():
	s=0	
	for i in [[f"{i}/{j}"for j in os.listdir(i)] for i in ["train_devil/train/0"]]:
		for i in i:
			print(i)
			s+=1
			cv2.imwrite(i,cv2.resize(cv2.imread(i),(200,200)))

"""def spliter_devil(data_size,data=pandas.read_csv("train_metadata.csv"),count=0,count_=0):
	
	[os.makedirs(i) for i in ['train_devil','train_devil/train/','train_devil/test/','train_devil/train/0','train_devil/train/1','train_devil/test/0','train_devil/test/1'] if os.path.exists(i)==False]
	for i in zip(data["isic_id"],data["target"]):
		if i[1]==0:
			if count<=data_size-300:
				os.popen(f'cp train-image/image/{i[0]}.jpg train_devil/train/0/{i[0]}.jpg')
				count+=1
			elif count<=data_size:
				os.popen(f'cp train-image/image/{i[0]}.jpg train_devil/test/0/{i[0]}.jpg')
				count+=1

		elif i[1]==1:
			if count_<=93:
				os.popen(f'cp train-image/image/{i[0]}.jpg train_devil/train/1/{i[0]}.jpg')
				count_+=1
			elif count_<=393:
				os.popen(f'cp train-image/image/{i[0]}.jpg train_devil/test/1/{i[0]}.jpg')
				count_+=1

"""
#spliter_devil(1590)
image_resizer()

#for j in os.listdir(r"/media/devil/New Volume/fake/face_2/Dataset/Train/Real"):
#	shutil.copy(f"/media/devil/New Volume/fake/face_2/Dataset/Train/Real/{j}",r"train_devil/train/0")
