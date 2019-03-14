# -*- coding: utf-8 -*-

from iss.models import AbstractAutoEncoderModel
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, BatchNormalization, Activation, Lambda
from keras.optimizers import Adadelta, Adam
from keras.models import Model
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np

class VarAutoEncoder(AbstractAutoEncoderModel):

	def __init__(self, config):

		save_directory = config['save_directory']
		model_name = config['model_name']

		super().__init__(save_directory, model_name)

		self.activation = config['activation']
		self.input_shape = (config['input_height'], config['input_width'], config['input_channel'])
		self.latent_shape = config['latent_shape']
		self.lr = config['learning_rate']
		self.build_model()

	def sampling(self, args):
		z_mean, z_log_var = args

		batch = K.shape(z_mean)[0]
		dim = K.int_shape(z_mean)[1]
		epsilon = K.random_normal(shape=(batch, dim))
		return z_mean + K.exp(0.5 * z_log_var) * epsilon

	def build_model(self):
		input_shape = self.input_shape
		latent_shape = self.latent_shape

		picture = Input(shape = input_shape)

		x = Flatten()(picture)
		x = Dense(1000, activation = 'relu', name = 'enc_1')(x)
		x = Dense(100, activation = 'relu', name = 'enc_2')(x)
		z_mean = Dense(latent_shape, name = 'enc_z_mean')(x)
		z_log_var = Dense(latent_shape, name = 'enc_z_log_var')(x)


		z = Lambda(self.sampling, name='enc_z')([z_mean, z_log_var])

		self.encoder_model = Model(picture, [z_mean, z_log_var, z], name = "encoder")
		
		latent_input = Input(shape = (latent_shape,), name = "enc_z_sampling")
		x = Dense(100, activation = 'relu', name = 'dec_1')(latent_input)
		x = Dense(1000, activation = 'relu', name = 'dec_2')(x)
		x = Dense(np.prod(input_shape), activation = self.activation)(x)
		decoded = Reshape((input_shape))(x)

		self.decoder_model = Model(latent_input, decoded, name = "decoder")

		picture_dec = self.decoder_model(self.encoder_model(picture)[2]) 
		self.model = Model(picture, picture_dec, name = "autoencoder")

		def my_loss(picture, picture_dec):

			xent_loss = K.mean(K.binary_crossentropy(picture, picture_dec), axis = (-1, -2, -3))
			kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
			loss =  K.mean(xent_loss + kl_loss)
			return loss

		optimizer = Adam(lr = self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
		
		self.model.compile(optimizer = optimizer, loss = my_loss)

