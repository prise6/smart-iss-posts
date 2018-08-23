import tensorflow as tf
import os
import sys
import numpy as np

class BaseTrainer:

	def __init__(self, sess, model, data_loader):

		self.sess = sess
		self.model = model
		self.data_loader = data_loader

		self.init = tf.global_variables_initializer()
		sess.run(self.init)

		self.model.load(self.sess)

	def train(self):
		self.data_loader.initialize(self.sess)
		for cur_epoch in range(self.model.cur_epoch.eval(self.sess), int(os.getenv('NUM_EPOCH')) + 1):
			ent_train, acc_train = self.train_epoch(cur_epoch)
			self.sess.run(self.model.increment_cur_epoch)
			ent_test, acc_test = self.test(cur_epoch)
			print("Epoch-{} ent:{:.4f} -- acc:{:.4f} | ent:{:.4f} -- acc:{:.4f}".format(
				cur_epoch, ent_train, acc_train, ent_test, acc_test))

			
	def train_epoch(self, epoch = None):

		self.data_loader.set_is_train(is_train = True)

		entropies = []
		accuracies = []

		for _ in range(int(os.getenv('NUM_ITER_BATCH'))):
			ent, acc = self.train_step()
			# print("acc : {}".format(acc))
			entropies.append(ent)
			accuracies.append(acc)

		ent = np.mean(entropies)
		acc = np.mean(accuracies)

		return ent, acc


		# self.model.save(self.sess)

	def train_step(self):
		batch_x, batch_y = self.data_loader.get_input(self.sess)
		feed_dict = {
			self.model.x: batch_x,
			self.model.y: batch_y,
			self.model.is_training: True
		}
		_, ent, acc, tmp = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy, self.model.tmp], feed_dict = feed_dict)
		# print("tmp : {}".format(tmp))
		# print("selfy : {}".format(batch_y))

		return ent, acc

	def test(self, epoch):
		self.data_loader.set_is_train(is_train = False)

		test_x, test_y = self.data_loader.get_input(self.sess)
		feed_dict = {
			self.model.x: test_x,
			self.model.y: test_y,
			self.model.is_training: False
		}
		ent, acc, tmp = self.sess.run([self.model.cross_entropy, self.model.accuracy, self.model.tmp], feed_dict = feed_dict)

		# print("Sur les donnees test {} - ent:{:.4f} -- acc:{:.4f}".format(epoch, ent, acc))
		# print("selfy : {}".format(test_y))

		return ent, acc