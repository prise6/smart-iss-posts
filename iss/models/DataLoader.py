# -*- coding: utf-8 -*-
import os

from keras.preprocessing.image import ImageDataGenerator

class ImageDataGeneratorWrapper:

	def __init__(self, config, model):
		
		self.config = config
		self.model = model
		self.datagen = None
		self.train_generator = None
		self.test_generator = None

		self.image_data_generator(config)

		sampling_type = self.config.get('models')[self.model]['sampling']
		train_dir = os.path.join(self.config.get('sampling')[sampling_type]['directory']['train'], '..')
		test_dir = os.path.join(self.config.get('sampling')[sampling_type]['directory']['test'], '..')

		self.set_train_generator(train_dir)
		self.set_test_generator(test_dir)

	def image_data_generator(self, config):
		self.datagen = ImageDataGenerator(
  			rescale = 1./255
		)
		return self

	def build_generator(self, directory):
		# voir plus tars si besoin de parametrer
		return self.datagen.flow_from_directory(
			directory,
			target_size = (self.config.get('models')[self.model]['input_height'], self.config.get('models')[self.model]['input_width']),
			color_mode = 'rgb',
			classes = None,
			class_mode = 'input',
			batch_size = self.config.get('models')[self.model]['batch_size'], 
		)

	def set_train_generator(self, train_dir):
		self.train_generator = self.build_generator(directory = train_dir)
		return self

	def get_train_generator(self):
		return self.train_generator

	def set_test_generator(self, test_dir):
		self.test_generator = self.build_generator(directory = test_dir)
		return self

	def get_test_generator(self):
		return self.train_generator



