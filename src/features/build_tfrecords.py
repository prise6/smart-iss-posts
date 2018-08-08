# -*- coding: utf-8 -*-
import click
import logging
import os
import sys
import pandas as pd
import tensorflow as tf
import numpy as np
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from PIL import Image


load_dotenv(find_dotenv())

def main():
    """ Create train and test test
    """

    logger.info('Create train & test set')

    # create refs labels
    refs = pd.read_csv(refs_path)
    refs = refs.pivot_table(index = "image", columns = "label", aggfunc = len, fill_value = 0).reset_index()
    refs.rename(columns = lambda x: "label" + str(x) if x != 'image' else x, inplace = True)
    refs = refs[refs['image'].isin(imgs_list)]

    # initiate tf file writer
    writer = tf.python_io.TFRecordWriter(train_filename)

    for i, row in refs.iterrows():
        img_full_path = os.path.join(imgs_path, row['image'])
        path, raw  = loadImg(img_full_path)

        example = tf.train.Example(features = tf.train.Features(feature = {
            'input' : _bytes_feature(raw),
            'label' : _int64_feature(row['label1'])
        }))

        logger.info("Creation TFRECORDS {}/{} : {}".format(i+1, refs.index.size, img_full_path))
        writer.write(example.SerializeToString())

    writer.close()


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def loadImg(path):

    raw = np.asarray(Image.open(path)).tostring()
    
    return path, raw


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    logger = logging.getLogger(__name__)
    imgs_path = os.path.join(str(project_dir), 'data', 'interim', 'collections')
    imgs_list = os.listdir(imgs_path)
    refs_path = os.path.join(str(project_dir), 'data', 'external', 'refs', 'references_labels.csv')
    train_filename = os.path.join(str(project_dir), 'data', 'processed', 'train.tfrecords')   

    main()
