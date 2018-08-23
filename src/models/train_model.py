import tensorflow as tf
import os
import sys
from dotenv import find_dotenv, load_dotenv
from data_loader import TFRecordsLoader
from trainer_model import BaseTrainer
from base_model import BaseModel


def main():
	load_dotenv(find_dotenv())

	sess = tf.Session()
	
	model = BaseModel()
	data_loader = TFRecordsLoader()

	trainer = BaseTrainer(sess, model, data_loader)
	trainer.train()

if __name__ == '__main__':
	main()
