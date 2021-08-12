import numpy as np
from scipy.misc import imread, imsave, imresize
import sys
import random
import os
import tensorflow as tf
import torch
sys.path.append('align_methods')

from align import align, re_align
from collections import OrderedDict
import time

def makedirs(path):
	if not os.path.exists(path):
		os.makdirs(path)

def save_priv(image, align_image, original_image, src, M, epsilon, model_name, output_dir):
	"""Saves images to the output directory.

	Args:
			images: array with minibatch of images, normalized
			align_image: unnormalized
			original_iamge: unnormalized
			filenames: list of filenames without path
					If number of file names in this list less than number of images in
					the minibatch then only first len(filenames) images will be saved.
			output_dir: directory where to save images
	"""
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	path = src + '-' + str(epsilon) + '.png'
	with tf.gfile.Open(os.path.join(output_dir, path), 'w') as f:
		image = imresize(image, (112, 112))
		align_image = imresize(align_image, (112, 112))
		image = re_align(image, align_image, original_image, M)
		image = np.clip(image, original_image - epsilon, original_image + epsilon)
		image = np.clip(image, 0, 255).astype(np.uint8)
		imsave(f, image.astype(np.uint8), format='png')
	return os.path.join(output_dir, path)
    
def cosdistance(x, y, offset=1e-5):
	x = x / torch.sqrt(torch.sum(x**2)) + offset
	y = y / torch.sqrt(torch.sum(y**2)) + offset
	return torch.sum(x * y)

def L2distance(x, y):
	return torch.sqrt(torch.sum((x - y)**2))

def inference(x, model, use_prelogits=False, image_shape=(112, 112)):
	"""
		get the embedding of face images
		:param x: ndarray(size = [None, None, None, 3]), unnormalized
		:param model: our model class
		:param image_shape: (int, int), the shape that a cropped image should be (related to networks, not images)
		return: torch.Tensor(size = [None, None])
	"""
	assert x.ndim == 3 or x.ndim == 4
	if x.ndim == 3:
		x = x[None, :]
	batchsize = x.shape[0]

	if x.shape[1:3] != (112, 112) and x.shape[1:3] != image_shape:
		align_x = np.zeros((batchsize, 112, 112, 3))
		for i in range(batchsize):
			align_x[i], _ = align(x[i])
		x = align_x
	if x.shape[1:3] != image_shape:
		normal_x = x
		x = np.zeros((batchsize, image_shape[0], image_shape[1], 3))
		for i in range(batchsize):
			x[i] = imresize(normal_x[i], (image_shape[0], image_shape[1]))
	x = torch.Tensor(x).cuda()
	x = x.permute(0, 3, 1, 2)
	return model.forward(x, use_prelogits)


def compare(model, img1_path, img2_path, image_shape):
	img1 = imread(img1_path, mode='RGB').astype(np.float32)
	img2 = imread(img2_path, mode='RGB').astype(np.float32)
	feat1 = inference(img1[np.newaxis], model, use_prelogits=False, image_shape=image_shape)
	feat2 = inference(img2[np.newaxis], model, use_prelogits=False, image_shape=image_shape)
	cos_similarity = cosdistance(feat1, feat2)

	feat1 = inference(img1[np.newaxis], model, use_prelogits=True, image_shape=image_shape)
	feat2 = inference(img2[np.newaxis], model, use_prelogits=True, image_shape=image_shape)
	L2_similarity = -L2distance(feat1, feat2)
	print(feat2[0, :10])
	return cos_similarity, L2_similarity

