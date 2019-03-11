# -*- coding: utf-8 -*-

from iss.models.AbstractModel import AbstractModel
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten
from keras.optimizers import Adadelta, Adam
from keras.models import Model
import numpy as np

class SimpleConvAutoEncoder(AbstractModel):

		def __init__(self, config):

			save_directory = config['save_directory']
			model_name = config['model_name']

			super().__init__(save_directory, model_name)

			self.activation = config['activation']
			self.input_shape = (config['input_height'], config['input_width'], config['input_channel'])
			self.lr = config['learning_rate']
			self.build_model()

		def build_model(self):
			input_shape = self.input_shape

			picture = Input(shape = input_shape)

			# encoded network
			x = Conv2D(4, (3, 3), activation = 'relu', padding = 'same', name = 'enc_conv_1')(picture)
			x = MaxPooling2D((2, 2))(x)
			x = Conv2D(8, (3, 3), activation = 'relu', padding = 'same', name = 'enc_conv_2')(x)
			encoded = MaxPooling2D((2, 2))(x)

			# decoded network
			x = Conv2D(8, (3, 3), activation = 'relu', padding = 'same', name = 'dec_conv_1')(encoded)
			x = UpSampling2D((2, 2))(x)
			x = Conv2D(4, (3, 3), activation = 'relu', padding = 'same', name = 'dec_conv_2')(x)
			x = UpSampling2D((2, 2))(x)
			x = Flatten()(x)
			x = Dense(np.prod(input_shape), activation = self.activation)(x)
			decoded = Reshape((input_shape))(x)

			self.model = Model(picture, decoded)
			
			# optimizer = Adadelta(lr = self.lr, rho = 0.95, epsilon = None, decay = 0.0)
			optimizer = Adam(lr = 0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
			
			self.model.compile(optimizer = optimizer, loss = 'binary_crossentropy')
