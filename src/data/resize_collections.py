# -*- coding: utf-8 -*-
import click
import logging
import os
import sys
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from PIL import Image

load_dotenv(find_dotenv())

@click.command()
@click.argument('RESIZE_WIDTH', type=int, default=os.getenv('RESIZE_WIDTH'))
@click.argument('RESIZE_HEIGHT', type=int, default=os.getenv('RESIZE_HEIGHT'))
def main(resize_width, resize_height):
    """ Resize image
    """

    logger.info('Resize collections to {}x{}'.format(resize_width, resize_height))

    try:
        imgs_path = os.path.join(str(project_dir), 'data', 'external', 'collections')
        [resize_one_img(os.path.join(imgs_path, img_path), resize_width, resize_height) for img_path in os.listdir(imgs_path)]
    except:
        logger.error(sys.exc_info()[0])
        exit()


def resize_one_img(img_path, resize_width, resize_height):
   
    logger.info('Resize {}'.format(img_path))
    size = (resize_width, resize_height)
    outfile = os.path.join(str(project_dir), 'data', 'interim', 'collections', os.path.basename(img_path))
    try:
        im = Image.open(os.path.join(str(project_dir), img_path))
        im.thumbnail(size)
        im.save(outfile, "JPEG")
    except IOError:
        logger.info('Cannot resize {}'.format(img_path))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    logger = logging.getLogger(__name__)

    main()
    