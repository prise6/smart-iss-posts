# -*- coding: utf-8 -*-

from keras.callbacks import ModelCheckpoint, CSVLogger
from iss.models.Callbacks import DisplayPictureCallback
from iss.tools.tools import Tools
import keras

class ModelTrainer:

	def __init__(self, model, data_loader, config, callbacks=[]):
		self.model = model
		self.data_loader = data_loader

		self.epochs = config['epochs']
		self.verbose = config['verbose']
		self.initial_epoch = config['initial_epoch']
		self.workers = config['workers']
		self.use_multiprocessing = config['use_multiprocessing']
		self.steps_per_epoch = config['steps_per_epoch']
		self.validation_steps = config['validation_steps']
		self.validation_freq = config['validation_freq']
		self.callbacks = callbacks

		self.init_callbacks(config)

	def train(self):

		print(keras.__version__)

		if self.validation_freq is not None:
			self.model.model.fit_generator(
				generator = self.data_loader.get_train_generator(),
				steps_per_epoch = self.steps_per_epoch, 
				epochs = self.epochs,
				verbose = self.verbose,
				initial_epoch = self.initial_epoch,
				callbacks = self.callbacks,
				workers = self.workers,
				use_multiprocessing = self.use_multiprocessing,
				validation_data = self.data_loader.get_test_generator(),
				validation_steps = self.validation_steps,
				validation_freq = self.validation_freq
			)
		else:
			self.model.model.fit_generator(
				generator = self.data_loader.get_train_generator(),
				steps_per_epoch = self.steps_per_epoch, 
				epochs = self.epochs,
				verbose = self.verbose,
				initial_epoch = self.initial_epoch,
				callbacks = self.callbacks,
				workers = self.workers,
				use_multiprocessing = self.use_multiprocessing,
				validation_data = self.data_loader.get_test_generator(),
				validation_steps = self.validation_steps
			)

		self.model.save()

	def init_callbacks(self, config):


		if 'csv_logger' in config['callbacks']:
			log_dir = config['callbacks']['csv_logger']['directory']
			Tools.create_dir_if_not_exists(log_dir)

			self.csv_logger = CSVLogger(
				filename = '{}/{}_training.log'.format(log_dir, self.model.model_name),
				append = config['callbacks']['csv_logger']['append']
			)
			self.callbacks.extend([self.csv_logger])

		if 'checkpoint' in config['callbacks']:
			chekpt_dir = config['callbacks']['checkpoint']['directory']
			Tools.create_dir_if_not_exists(chekpt_dir)
			self.checkpointer = ModelCheckpoint(
				filepath = chekpt_dir + '/' + self.model.model_name + '-{epoch:02d}.hdf5',
				verbose = config['callbacks']['checkpoint']['verbose'],
				period = config['callbacks']['checkpoint']['period']
			)
			self.callbacks.extend([self.checkpointer])

		if 'display_picture' in config['callbacks']:
			self.picture_displayer = DisplayPictureCallback(
				model = self.model,
				data_loader = self.data_loader.get_train_generator(),
				epoch_laps = config['callbacks']['display_picture']['epoch_laps']
			)
			self.callbacks.extend([self.picture_displayer])
		
		if 'tensorboard' in config['callbacks']:
			log_dir = config['callbacks']['tensorboard']['log_dir']
			Tools.create_dir_if_not_exists(log_dir)
			self.callbacks.extend([keras.callbacks.TensorBoard(
				log_dir = log_dir,
				histogram_freq=0,
				batch_size=32,
				write_graph=False,
				write_images = True
			)])


		return self
