# -*- coding: utf-8 -*-

from iss.tools import Config
from iss.tools import Tools
from iss.models import SimpleConvAutoEncoder
from iss.clustering import ClassicalClustering
from dotenv import find_dotenv, load_dotenv

## Config
load_dotenv(find_dotenv())
cfg = Config(project_dir = os.getenv("PROJECT_DIR"), mode = os.getenv("MODE"))

## charger le modèle
model_type = 'simple_conv'
cfg.get('models')[model_type]['model_name'] = 'model_colab'
model = SimpleConvAutoEncoder(cfg.get('models')[model_type])

## Générateur d'image
filenames = Tools.list_directory_filenames('data/processed/models/autoencoder/train/k/')
generator_imgs = Tools.generator_np_picture_from_filenames(filenames, target_size = (27, 48), batch = 496, nb_batch = 2)

## Générer des images