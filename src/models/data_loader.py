import tensorflow as tf
import os
import sys


class TFRecordsLoader:
    """
        DataSetAPI - Load TFRecords from the disk
    """

    def __init__(self):

        self.is_train = None

        self.dataset = tf.data.TFRecordDataset(os.getenv('TRAIN_TFRECORD'))
        self.dataset = self.dataset.map(TFRecordsLoader.parser)
        self.dataset = self.dataset.shuffle(1000)
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.batch(int(os.getenv("BATCH_SIZE")))

        self.test = tf.data.TFRecordDataset(os.getenv('TEST_TFRECORDS'))
        self.test = self.test.map(TFRecordsLoader.parser)
        self.test = self.test.repeat()
        self.test = self.test.batch(int(os.getenv("BATCH_SIZE")))

        self.train_it = self.dataset.make_one_shot_iterator().string_handle()
        self.test_it = self.test.make_one_shot_iterator().string_handle()

        self.handle = tf.placeholder(tf.string, shape=[])

        self.iterator = tf.data.Iterator.from_string_handle(self.handle, self.dataset.output_types, self.dataset.output_shapes)
        

    @staticmethod
    def parser(record):
        keys_to_features = {
            'input': tf.FixedLenFeature((), tf.string),
            'label': tf.FixedLenFeature((), tf.int64)
        }

        parsed = tf.parse_single_example(record, keys_to_features)
        image = tf.decode_raw(parsed['input'], tf.float64)
        image = tf.reshape(image, [36, 64, 3])
        image = tf.cast(image, tf.float32)
        label = parsed['label']

        return image, label

    def set_is_train(self, is_train):
        self.is_train = is_train

    def initialize(self, sess):
        self.train_handle, self.test_handle = sess.run([self.train_it, self.test_it])

    def get_input(self, sess):
        return sess.run(self.iterator.get_next(), feed_dict = {self.handle: self.train_handle if self.is_train else self.test_handle})        
