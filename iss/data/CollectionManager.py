# -*- coding: utf-8 -*-

import os
import random
import shutil
import numpy as np
import re

class CollectionManagerFromDirectory:

	def __init__(self, dir, config):
		self.config = config

		jpg_regex = re.compile(".*jpg$")
		self.pictures_id = [pict for pict in os.listdir(dir) if jpg_regex.match(pict)]
		
		self.dir = os.path.join(self.config.project_dir, dir)
		self.dir_base = os.path.join(self.config.project_dir, self.config.get('directory')['autoencoder']['base'])
		self.dir_train = os.path.join(self.config.project_dir, self.config.get('directory')['autoencoder']['train'])
		self.dir_test = os.path.join(self.config.project_dir, self.config.get('directory')['autoencoder']['test'])
		self.dir_valid = os.path.join(self.config.project_dir, self.config.get('directory')['autoencoder']['valid'])
		
		self.seed = self.config.get('training')['seed']
		self.proportions = self.config.get('training')['proportions']
		self.volumes = {}

		self.shuffle()

	def count(self):
		self.volumes['total'] = len(self.pictures_id)
		return self

	def shuffle(self):
		random.seed(self.seed)
		random.shuffle(self.pictures_id)
		return self

	@staticmethod
	def create_dir(path):
		if os.path.exists(path):
			shutil.rmtree(path, ignore_errors=True)

		return os.makedirs(path)

	@staticmethod
	def copy_pictures(dest, picture_dir, pictures_id):

		pictures_src = os.path.join(picture_dir, pictures_id)
		pictures_dest = os.path.join(dest, pictures_id)
		return shutil.copyfile(pictures_src, pictures_dest)

	def createDirectories(self):
		CollectionManagerFromDirectory.create_dir(self.dir_base)
		CollectionManagerFromDirectory.create_dir(self.dir_train)
		CollectionManagerFromDirectory.create_dir(self.dir_test)
		CollectionManagerFromDirectory.create_dir(self.dir_valid)

		return self


	def populateDirectories(self):
		self.volumes['train'] = int(np.floor(self.proportions['train'] * self.volumes['total']))
		self.volumes['test'] = int(np.floor(self.proportions['test'] * self.volumes['total']))
		self.volumes['valid'] = self.volumes['total'] - (self.volumes['train'] + self.volumes['test'])

		for pict in self.pictures_id[:self.volumes['train']]:
			CollectionManagerFromDirectory.copy_pictures(self.dir_train, self.dir, pict)

		for pict in self.pictures_id[self.volumes['train']:(self.volumes['train'] + self.volumes['test'])]:
			CollectionManagerFromDirectory.copy_pictures(self.dir_test, self.dir, pict)

		for pict in self.pictures_id[-self.volumes['valid']:]:
			CollectionManagerFromDirectory.copy_pictures(self.dir_valid, self.dir, pict)





