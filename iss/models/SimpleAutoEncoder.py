# -*- coding: utf-8 -*-

from iss.models import AbstractAutoEncoderModel
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten
from keras.optimizers import Adadelta, Adam
from keras.models import Model
import numpy as np

class SimpleAutoEncoder(AbstractAutoEncoderModel):

		def __init__(self, config):

			save_directory = config['save_directory']
			model_name = config['model_name']

			super().__init__(save_directory, model_name)

			self.activation = config['activation']
			self.input_shape = (config['input_height'], config['input_width'], config['input_channel'])
			self.latent_shape = config['latent_shape']
			self.lr = config['learning_rate']
			self.build_model()

		def build_model(self):
			input_shape = self.input_shape

			picture = Input(shape = input_shape)

			# encoded network
			x = Flatten()(picture)
			layer_1 = Dense(1000, activation = 'relu', name = 'enc_1')(x)
			layer_2 = Dense(100, activation = 'relu', name = 'enc_2')(layer_1)
			encoded = Dense(self.latent_shape, activation = 'relu', name = 'enc_3')(layer_2)
			self.encoder_model = Model(picture, encoded, name = "encoder")

			#  decoded netword
			latent_input = Input(shape = (self.latent_shape,))
			layer_4 = Dense(100, activation = 'relu', name = 'dec_1')(latent_input)
			layer_5 = Dense(1000, activation = 'relu', name = 'dec_2')(layer_4)

			x = Dense(np.prod(input_shape), activation = self.activation)(layer_5)
			decoded = Reshape((input_shape))(x)

			self.decoder_model = Model(latent_input, decoded, name = "decoder")
			
			picture_dec = self.decoder_model(self.encoder_model(picture))
			self.model = Model(picture, picture_dec)
			
			# optimizer = Adadelta(lr = self.lr, rho = 0.95, epsilon = None, decay = 0.0)
			optimizer = Adam(lr = 0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
			
			self.model.compile(optimizer = optimizer, loss = 'binary_crossentropy')
