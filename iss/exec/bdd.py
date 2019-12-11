import os
import time
import mysql.connector
import pandas as pd
import datetime as dt
import numpy as np

from iss.init_config import CONFIG
from iss.data.DataBaseManager import MysqlDataBaseManager
from iss.clustering import ClassicalClustering, AdvancedClustering, N2DClustering
from iss.tools import Tools


def create_db_manager(config):

    CON_MYSQL = mysql.connector.connect(
        host = config.get('mysql')['database']['server'],
        user = config.get('mysql')['database']['user'],
        passwd = config.get('mysql')['database']['password'],
        database = config.get('mysql')['database']['name'],
        port = config.get('mysql')['database']['port']
    )

    return MysqlDataBaseManager(CON_MYSQL, config)


def populate_locations(config, db_manager):

    history = pd.read_csv(os.path.join(CONFIG.get("directory")['data_dir'], "raw", "history", "history.txt"), sep=";", names=['latitude', 'longitude', 'id', 'location'])
    history['timestamp'] = pd.to_datetime(history.id, format="%Y%m%d-%H%M%S").dt.strftime("%Y-%m-%d %H:%M:%S")
    history.fillna('NULL', inplace=True)
    history = history[['latitude', 'longitude', 'id', 'timestamp', 'location']]
    history_tuple = [tuple(x) for x in history.values]

    db_manager.create_pictures_location_table(force=True)
    count = db_manager.insert_row_pictures_location(history_tuple)

    print("Nombre d'insertion: %s" % count)


def populate_embedding(config, db_manager, clustering_type, clustering_version, clustering_model_type, clustering_model_name, drop=False):

    db_manager.create_pictures_embedding_table()
    clustering_config = config.get('clustering')[clustering_type]
    clustering_config['version'] = clustering_version
    clustering_config['model']['type'] = clustering_model_type
    clustering_config['model']['name'] = clustering_model_name

    if drop:
        db_manager.drop_embedding_partition(clustering_type, clustering_version, clustering_model_type, clustering_model_name)

    if clustering_type == 'n2d':
        clustering = N2DClustering(clustering_config)
    elif clustering_type == 'classical':
        clustering = ClassicalClustering(clustering_config)
    else:
        raise Exception

    clustering.load()
    model, model_config = Tools.load_model(CONFIG, clustering_model_type, clustering_model_name)
    filenames = Tools.list_directory_filenames(CONFIG.get('directory')['collections'])
    generator = Tools.load_latent_representation(CONFIG, model, model_config, filenames, 496, None, True)

    count = 0
    for ids, latents in generator:
        pictures_embedding = clustering.predict_embedding(latents)
        rows = []
        for i, id in enumerate(ids):
            rows.append((
                id,
                float(np.round(pictures_embedding[i][0], 4)),
                float(np.round(pictures_embedding[i][1], 4)),
                clustering_type,
                clustering_version,
                clustering_model_type,
                clustering_model_name
            ))
        count += db_manager.insert_row_pictures_embedding(rows)
        print("Nombre d'insertion: %s / %s" % (count, len(filenames)))


    return


def main(action = 'populate_embedding'):

    db_manager = create_db_manager(CONFIG)

    if action == 'population_locations':
        populate_locations(CONFIG, db_manager)
    elif action == 'populate_embedding':
        populate_embedding(CONFIG, db_manager, 'n2d', 1, 'simple_conv', 'model_colab')
    else:
        pass


if __name__ == '__main__':
    main()