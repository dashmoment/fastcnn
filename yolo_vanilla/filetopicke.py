from __future__ import print_function

import numpy as np
import tensorflow as tf
import voc_utils
import os
from shutil import copyfile
import pickle

def voc_savefiletopickle(root_dir, pickfile, category, annotatefile):

	pathlist = []
	cat_all = voc_utils.list_image_sets()

	with open(pickfile, 'ab') as handle:
		for cat in cat_all:
		
			imlist = voc_utils.imgs_from_category_as_list(category,cat,annotatefile)
			#print(cat)
			#print(root_dir)
			#print(imlist)
			for im in imlist:

				
				imgpath = os.path.join(root_dir, im+".jpg")
				pathlist.append(imgpath)

		pickle.dump(pathlist, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	handle.close()

root_dir12 = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2012'
root_dir07 = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2007'

ann_dir12 = os.path.join(root_dir12, 'ImageSets/Main/')
ann_dir07 = os.path.join(root_dir07, 'ImageSets/Main/')
tann_dir12 = os.path.join(root_dir12, 'Test/ImageSets/Main/')
tann_dir07 = os.path.join(root_dir07, 'Test/ImageSets/Main/')

img_dir12 = os.path.join(root_dir12, 'JPEGImages/')
img_dir07 = os.path.join(root_dir07, 'JPEGImages/')

timg_dir12 = os.path.join(root_dir12, 'Test/JPEGImages/')
timg_dir07 = os.path.join(root_dir07, 'Test/JPEGImages/')

srcdir = {'train':[img_dir12,img_dir07],'test': [timg_dir07]}
anndir = {'train':[ann_dir12,ann_dir07],'test':[tann_dir07]}
catlist = ['train', 'val']

for i in range(len(srcdir['train'])):
	
	for cat in catlist:
		voc_savefiletopickle (srcdir['train'][i],'train.pickle', cat,anndir['train'][i])

#for img_dir in srcdir['test'],:
#	
#		voc_savefiletopickle (img_dir,'train.pickle', 'test')


with open('train.pickle', 'rb') as handle:

	objs = []
	while 1:
	    try:
	        objs = objs + pickle.load(handle)
	    except EOFError:
	        break
	#print(len(objs))
handle.close()