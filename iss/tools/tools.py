# -*- coding: utf-8 -*-

import PIL
import os

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


