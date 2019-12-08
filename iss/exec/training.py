import os
import click

from iss.init_config import CONFIG
from iss.models.DataLoader import ImageDataGeneratorWrapper
from iss.models.ModelTrainer import ModelTrainer
from iss.models import SimpleAutoEncoder
from iss.models import SimpleConvAutoEncoder
from iss.models import VarAutoEncoder
from iss.models import VarConvAutoEncoder


@click.command()
@click.option('--model-type', default='simple_conv', show_default=True, type=str)
@click.option('--load', default=False, is_flag=True)
@click.option('--load-name', default=None, show_default=True, type=str)
def main(model_type, load, load_name):

    ## Variables globales
    _MODEL_TYPE = model_type
    _LOAD_NAME = load_name
    _LOAD = load

    ## Data loader
    data_loader = ImageDataGeneratorWrapper(CONFIG, model = _MODEL_TYPE)

    ## Model
    if _MODEL_TYPE in ['simple_conv']:
        model = SimpleConvAutoEncoder(CONFIG.get('models')[_MODEL_TYPE])
    elif _MODEL_TYPE in ['simple']:
        model = SimpleAutoEncoder(CONFIG.get('models')[_MODEL_TYPE])
    else:
        raise Exception

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


if __name__ == '__main__':
    main()
