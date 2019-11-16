# -*- coding: utf-8 -*-

import PIL
import os
import numpy as np
from io import BytesIO
import base64
from scipy.cluster.hierarchy import dendrogram
from keras_preprocessing.image.utils import load_img
import matplotlib as plt


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
	def display_mosaic(array, nrow = 5, ncol_max = 10):

		tmp = []
		i = 0
		image_col = []
		while i < len(array):
			tmp.append(array[i])

			if len(tmp) % nrow == 0 and i > 0:
				image_col.append(np.concatenate(tuple(tmp)))
				tmp = []
				if len(image_col) == ncol_max:
					break
			i += 1
		if not image_col:
			image_col.append(np.concatenate(tuple(tmp)))
		image = np.concatenate(tuple(image_col), axis = 1)
		return Tools.display_one_picture(image)

	@staticmethod
	def create_dir_if_not_exists(path):
		if not os.path.exists(path):
			os.makedirs(path)
		return path

	@staticmethod
	def encoded_pictures_from_generator(generator, model):

		predictions_list = []
		predictions_id = []
		for imgs in generator:
			predictions_id.append(imgs[0])
			predictions_list.append(model.get_encoded_prediction(imgs[1]))
			
		predictions = np.concatenate(tuple(predictions_list), axis = 0)
		predictions_id = [os.path.splitext(os.path.basename(id))[0] for sub_id in predictions_id for id in sub_id]

		return predictions_id, predictions

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
		np.random.seed(33213)
		np.random.shuffle(filenames)
		filenames = [os.path.join(path,f) for f in filenames]

		return filenames

	@staticmethod
	def generator_np_picture_from_filenames(filenames, target_size = None, scale = 1, batch = 124, nb_batch = None):

		max_n = len(filenames)
		div = np.divmod(max_n, batch)

		if nb_batch is None:
			nb_batch = div[0] + 1 * (div[1] != 0)

		for i in range(nb_batch):
			i_debut = i*batch
			i_fin = min(i_debut + batch, max_n)
			yield (filenames[i_debut:i_fin], np.array([Tools.read_np_picture(f, target_size, scale) for f in filenames[i_debut:i_fin]]))

	@staticmethod
	def bytes_image(array):
		image = Tools.display_one_picture(array)
		buffer = BytesIO()
		image.save(buffer, format='png')
		im_bytes = buffer.getvalue()

		return im_bytes

	@staticmethod
	def base64_image(array):
		for_encoding = Tools.bytes_image(array)
		return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()

	@staticmethod
	def get_color_from_label(label, n_labels = 50, palette = 'viridis'):
		cmap = plt.cm.get_cmap(palette, n_labels)
		return plt.colors.to_hex(cmap(int(label)))


	@staticmethod
	def plot_dendrogram(model, **kwargs):

		# Children of hierarchical clustering
		children = model.children_

		# Distances between each pair of children
		# Since we don't have this information, we can use a uniform one for plotting
		distance = np.arange(children.shape[0])

		# The number of observations contained in each cluster level
		no_of_observations = np.arange(2, children.shape[0]+2)

		# Create linkage matrix and then plot the dendrogram
		linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

		# Plot the corresponding dendrogram
		dendrogram(linkage_matrix, **kwargs)