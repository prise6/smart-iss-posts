import os

from iss.init_config import CONFIG
from iss.models.DataLoader import ImageDataGeneratorWrapper
from iss.models.ModelTrainer import ModelTrainer
from iss.models import SimpleAutoEncoder
from iss.models import SimpleConvAutoEncoder
from iss.models import VarAutoEncoder
from iss.models import VarConvAutoEncoder

## Variables globales
_MODEL_TYPE = 'simple_conv'
_LOAD_NAME = None
_LOAD = False

## Data loader
data_loader = ImageDataGeneratorWrapper(CONFIG, model = _MODEL_TYPE)

## Model
if _MODEL_TYPE in ['simple_conv']:
    model = SimpleConvAutoEncoder(CONFIG.get('models')[_MODEL_TYPE])
    if _LOAD:
        model.load(which = _LOAD_NAME)
    model.encoder_model.summary()
    model.decoder_model.summary()

model.model.summary()

## Entraineur
trainer = ModelTrainer(model, data_loader, CONFIG.get('models')[_MODEL_TYPE], callbacks=[])

## Entrainement
try:
    trainer.train()
except KeyboardInterrupt:
    trainer.model.save()
