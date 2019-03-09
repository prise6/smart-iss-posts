# -*- coding: utf-8 -*-

from keras.callbacks import Callback
import numpy as np
from iss.tools.tools import Tools
from IPython.display import display

class DisplayPictureCallback(Callback):

	def __init__(self, model, epoch_laps, data_loader):

		self.model_class = model
		self.epoch_laps = epoch_laps
		self.data_loader = data_loader
		super(DisplayPictureCallback, self).__init__()


	def on_epoch_end(self, epoch, logs):
		if epoch % self.epoch_laps == 0:

			print("ok")

			input_pict = self.data_loader.next()[0][1]
			output_pict = self.model_class.predict_one(input_pict)

			display(Tools.display_one_picture_scaled(input_pict))
			display(Tools.display_index_picture_scaled(output_pict))
		
		return self


