# -*- coding: utf-8 -*-

from keras.models import load_model
import numpy as np
import os

class AbstractModel:
	def __init__(self, save_directory, model_name):
		self.save_directory = save_directory
		self.model = None
		self.model_name = model_name

	def save(self):
		if not os.path.exists(self.save_directory):
			os.makedirs(self.save_directory)

		self.model.save('{}/final_{}.hdf5'.format(self.save_directory, self.model_name))


	def load(self, which = 'final_model'):
		self.model = load_model('{}/{}.hdf5'.format(self.save_directory, which))

	def predict(self, x, batch_size = None, verbose = 0, steps = None, callbacks = None):
		return self.model.predict(x, batch_size, verbose, steps)

	def predict_one(self, x, batch_size = 1, verbose = 0, steps = None):
		x = np.expand_dims(x, axis = 0)
		return self.predict(x, batch_size, verbose, steps)


class AbstractAutoEncoderModel(AbstractModel):

	def __init__(self, save_directory, model_name):
		super().__init__(save_directory, model_name)
		self.encoder_model = None
		self.decoder_model = None

	def get_encoded_prediction(self, pictures):
		return self.encoder_model.predict(pictures)

	def get_full_encoded_prediction(self, generator, nb_batch = None):

		generator.reset()
		div = np.divmod(generator.n, generator.batch_size)
		
		if nb_batch is None:
			nb_batch = div[0] + 1 * (div[1] != 0) - 1

		if nb_batch <= 0:
			return

		predictions = self.get_encoded_prediction(generator.next()[1])
		while generator.batch_index <= (nb_batch - 1):
			predictions = np.concatenate((predictions, self.get_encoded_prediction(generator.next()[1]) ), axis = 0)
		
		return predictions

