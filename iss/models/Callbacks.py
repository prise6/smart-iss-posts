# -*- coding: utf-8 -*-

import os
import datetime
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from IPython.display import display

from iss.tools.tools import Tools


class DisplayPictureCallback(Callback):

    def __init__(self, model, epoch_laps, data_loader):

        self.model_class = model
        self.epoch_laps = epoch_laps
        self.data_loader = data_loader
        super(DisplayPictureCallback, self).__init__()


    def on_epoch_end(self, epoch, logs):
        if epoch % self.epoch_laps == 0:

            input_pict = self.data_loader.next()[0][1]
            output_pict = self.model_class.predict_one(input_pict)

            display(Tools.display_one_picture_scaled(input_pict))
            display(Tools.display_index_picture_scaled(output_pict))
        
        return self

class TensorboardCallback(Callback):

    def __init__(self, log_dir, limit_image = 1, model = None, data_loader = None):
        self.log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.limit_image = limit_image
        self.model_class = model
        self.data_loader = data_loader
        self.writer = tf.summary.FileWriter(self.log_dir)
        super(TensorboardCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        image_summaries = []
        
        for input_pict in self.data_loader.next()[0][:self.limit_image]:
            output_pict = self.model_class.predict_one(input_pict)[0]
            input_im_bytes = Tools.bytes_image(input_pict*255)
            output_im_bytes = Tools.bytes_image(output_pict*255)

            image_summaries.append(tf.Summary.Value(tag = 'input', image = tf.Summary.Image(encoded_image_string = input_im_bytes)))
            image_summaries.append(tf.Summary.Value(tag = 'output', image = tf.Summary.Image(encoded_image_string = output_im_bytes)))


        image_summary = tf.Summary(value = image_summaries)
        self.writer.add_summary(image_summary, epoch) 
        self._write_logs(logs, epoch)

        return self

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, index)
        self.writer.flush()


class FloydhubTrainigMetricsCallback(Callback):
    """FloydHub Training Metric Integration"""
    def on_epoch_end(self, epoch, logs=None):
        """Print Training Metrics"""
        print('{{"metric": "loss", "value": {}, "epoch": {}}}'.format(logs.get('loss'), epoch))
        print('{{"metric": "val_loss", "value": {}, "epoch": {}}}'.format(logs.get('val_loss'), epoch))

