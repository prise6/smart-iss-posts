# -*- coding: utf-8 -*-

import PIL
import os
import numpy as np
from keras_preprocessing.image.utils import load_img


class Tools:

	@staticmethod
	def display_one_picture(array):
		array = array.astype('uint8')
		return PIL.Image.fromarray(array, 'RGB')

	@staticmethod
	def display_one_picture_scaled(array):
		array = array * 255
		return Tools.display_one_picture(array)

	@staticmethod
	def display_index_picture_scaled(array, index = 0):
		return Tools.display_one_picture_scaled(array[index])

	@staticmethod
	def display_index_picture(array, index = 0):
		return Tools.display_one_picture(array[index])

	@staticmethod
	def create_dir_if_not_exists(path):
		if not os.path.exists(path):
			os.makedirs(path)
		return path

	@staticmethod
	def encoded_pictures_from_generator(generator, model):

		predictions_tuple = tuple([model.get_encoded_prediction(imgs) for imgs in generator])
		predictions = np.concatenate(predictions_tuple, axis = 0)

		return predictions

	@staticmethod
	def read_np_picture(path, target_size = None, scale = 1):
		# img = PIL.Image.open(filename)
		img = load_img(path, target_size = target_size)
		img_np = np.asarray(img, dtype = 'uint8')
		img_np = img_np * scale
		return img_np

	@staticmethod
	def list_directory_filenames(path):
		filenames = os.listdir(path)
		filenames = [path + f for f in filenames]

		return filenames

	@staticmethod
	def generator_np_picture_from_filenames(filenames, target_size = None, scale = 1, batch = 124, nb_batch = None):

		max_n = len(filenames)
		div = np.divmod(max_n, batch)

		if nb_batch is None:
			nb_batch = div[0] + 1 * (div[1] != 0)

		for i in range(nb_batch):
		# for i in [75, 76]:
			i_debut = i*batch
			i_fin = min(i_debut + batch, max_n)
			print("i_debut:" + str(i_debut))
			print("i_fin:" + str(i_fin))
			yield np.array([Tools.read_np_picture(f, target_size, scale) for f in filenames[i_debut:i_fin]])






