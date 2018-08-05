# -*- coding: utf-8 -*-
import click
import logging
import os
import sys
import pandas as pd
from shutil import copyfile
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

@click.command()
@click.argument('R_COLLECTIONS_PROJECT', type=click.Path(exists=True), default=os.getenv('R_COLLECTIONS_PROJECT'))
def main(r_collections_project):
    """ Synchronize my labeled image from another project with this one
    """

    logger.info('Synchronize labeled images')

    try:
        imgs = get_unique_imgs(r_collections_project)
        cp_imgs(r_collections_project, imgs)
    except:
        logger.error(sys.exc_info()[0])
        exit()

    return(1)

def get_unique_imgs(r_collections_project):
   
    logger.info('Copy reference file')
    copyfile(
        os.path.join(r_collections_project, 'datas', 'Export', 'references_labels.csv'),
        os.path.join(str(project_dir), "data", "external", "refs", "references_labels.csv")
    )
    refs = pd.read_csv(os.path.join(str(project_dir), 'data', 'external', 'refs', 'references_labels.csv'))
    imgs = refs.image.unique()

    return(imgs)


def cp_imgs(r_collections_project, imgs):

    logger.info('Synchronize images')
    
    img_path = os.path.join(r_collections_project, 'datas', 'Collections')

    i = 0
    img_len = len(imgs)

    for img in imgs:
        i += 1
        logger.info('Synchronize image {} {}/{}'.format(img, i, img_len))
        if(os.path.isfile(os.path.join(img_path, img))):
            copyfile(
                os.path.join(img_path, img),
                os.path.join(str(project_dir), "data", "external", "collections", img)
            )

    return(1)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    logger = logging.getLogger(__name__)

    main()
