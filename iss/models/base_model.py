import tensorflow as tf
import os
import sys
from dotenv import find_dotenv, load_dotenv
from data_loader import TFRecordsLoader
from trainer_model import BaseTrainer


class BaseModel:

	def __init__(self):
		
		self.cur_epoch = None
		self.increment_cur_epoch = None

		self.global_step = None
		self.increment_global_step = None

		self.init_global_step()
		self.init_cur_epoch()

		self.is_training = None
		self.x = None
		self.y = None

		self.cross_entropy = None
		self.accuracy = None
		self.train_step = None
		self.tmp = None

		self.build_model()
		self.init_saver()

	def save(self, sess):
		print("Saving model...")
		self.saver.save(sess, os.getenv('CHECKPOINT_DIR'), self.global_step)

	def load(self, sess):
		print("Load model...")
		latest_checkpoint = tf.train.latest_checkpoint(os.getenv('CHECKPOINT_DIR'))
		if latest_checkpoint:
			print("Loading model checkpoint {}...".format(latest_checkpoint))
			self.saver.restore(sess, latest_checkpoint)
			print("Model loaded")

	def init_cur_epoch(self):
		self.cur_epoch = tf.Variable(0, trainable = False)
		self.increment_cur_epoch = tf.assign(self.cur_epoch, self.cur_epoch + 1)

	def init_global_step(self):
		self.global_step = tf.Variable(0, trainable = False)
		self.increment_global_step = tf.assign(self.global_step, self.global_step + 1)

	def init_saver(self):
		self.saver = tf.train.Saver(max_to_keep = int(os.getenv('MAX_TO_KEEP')), save_relative_paths = True)

	def build_model(self):

		self.x = tf.placeholder(tf.float32, [None, 36, 64, 3])
		self.y = tf.placeholder(tf.int64, [int(os.getenv('BATCH_SIZE'))])


		self.is_training = tf.placeholder(tf.bool)


		conv1 = tf.layers.conv2d(self.x, 32, 5, activation = tf.nn.relu)
		conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

		conv2 = tf.layers.conv2d(conv1, 64, 3, activation = tf.nn.relu)
		conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

		fc1 = tf.contrib.layers.flatten(conv2)
		fc1 = tf.layers.dense(fc1, 98)
		fc2 = tf.layers.dense(fc1, 49)
		out = tf.layers.dense(fc2, 2)

		# labels = tf.reshape(tf.cast(self.y, tf.float32), (int(os.getenv('BATCH_SIZE')), -1))

		# self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = tf.nn.sigmoid(out)))
		# self.cross_entropy = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(labels, logits = tf.nn.sigmoid(out)))

		# # self.tmp = tf.count_nonzero(self.y)
		# # self.tmp = tf.reduce_min(tf.nn.softmax(out), axis = 1)
		# self.tmp = tf.nn.sigmoid(out)
		# self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.greater(tf.nn.sigmoid(out), 0.8), tf.int64), self.y), tf.float32))

		# self.train_step = tf.train.AdamOptimizer(learning_rate = 1e-3).minimize(self.cross_entropy)


		# avec une sortie de deux neurones:

		self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.y, logits = out))

		self.tmp = tf.nn.sigmoid(out)
		correct_prediction = tf.equal(tf.argmax(out, 1), self.y)
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		self.train_step = tf.train.AdamOptimizer(learning_rate = 1e-3).minimize(self.cross_entropy)

