import os

from iss.init_config import CONFIG
from iss.data.CollectionManager import CollectionManagerFromDirectory


## Variables globales
_SAMPLING_TYPE = 'autoencoder'

## Collection Manager
collection = CollectionManagerFromDirectory(config = CONFIG, sampling_type = _SAMPLING_TYPE)

## Volumes des images
volumes = collection.count().volumes
print(volumes)

## Creation des repertoires
collection.populateDirectories()