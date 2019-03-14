# -*- coding: utf-8 -*-

from iss.models import AbstractAutoEncoderModel
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, BatchNormalization, Activation
from keras.optimizers import Adadelta, Adam
from keras.models import Model
from keras import backend as K
import numpy as np

class SimpleConvAutoEncoder(AbstractAutoEncoderModel):

		def __init__(self, config):

			save_directory = config['save_directory']
			model_name = config['model_name']

			super().__init__(save_directory, model_name)

			self.activation = config['activation']
			self.input_shape = (config['input_height'], config['input_width'], config['input_channel'])
			self.latent_shape = (config['latent_height'], config['latent_width'], config['latent_channel'])
			self.lr = config['learning_rate']
			self.build_model()

		def build_model(self):
			input_shape = self.input_shape
			latent_shape = self.latent_shape

			picture = Input(shape = input_shape)

			# encoded network
			x = Conv2D(64, (3, 3), padding = 'same', name = 'enc_conv_1')(picture)
			x = BatchNormalization()(x)
			x = Activation('relu')(x)
			x = MaxPooling2D((2, 2))(x)

			x = Conv2D(32, (3, 3), padding = 'same', name = 'enc_conv_2')(x)
			x = BatchNormalization()(x)
			x = Activation('relu')(x)
			x = MaxPooling2D((2, 2))(x)

			x = Conv2D(16, (3, 3), padding = 'same', name = 'enc_conv_3')(x)
			x = BatchNormalization()(x)
			x = Activation('relu')(x)
			encoded = MaxPooling2D((2, 2))(x)

			self.encoder_model = Model(picture, encoded, name = "encoder")

			# decoded network
			latent_input = Input(shape = latent_shape)

			x = Conv2D(16, (3, 3), padding = 'same', name = 'dec_conv_1')(latent_input)
			x = BatchNormalization()(x)
			x = Activation('relu')(x)
			x = UpSampling2D((2, 2))(x)

			x = Conv2D(32, (3, 3), padding = 'same', name = 'dec_conv_2')(x)
			x = BatchNormalization()(x)
			x = Activation('relu')(x)
			x = UpSampling2D((2, 2))(x)

			x = Conv2D(64, (3, 3), padding = 'same', name = 'dec_conv_3')(x)
			x = BatchNormalization()(x)
			x = Activation('relu')(x)
			x = UpSampling2D((2, 2))(x)

			x = Conv2D(3, (3, 3), padding = 'same', name = 'dec_conv_4')(x)
			x = BatchNormalization()(x)
			x = Flatten()(x)
			x = Dense(np.prod(input_shape), activation = self.activation)(x)
			decoded = Reshape((input_shape))(x)

			self.decoder_model = Model(latent_input, decoded, name = "decoder")

			picture_dec = self.decoder_model(self.encoder_model(picture))
			self.model = Model(picture, picture_dec)
			
			optimizer = Adam(lr = self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
			
			def my_loss(picture, picture_dec):
				loss = K.mean(K.binary_crossentropy(picture, picture_dec))
				return loss

			# self.model.compile(optimizer = optimizer, loss = 'binary_crossentropy')
			self.model.compile(optimizer = optimizer, loss = my_loss)
