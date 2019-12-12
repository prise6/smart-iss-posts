import os
import time
import mysql.connector
import pandas as pd
import datetime as dt
import numpy as np

from iss.init_config import CONFIG
from iss.clustering import ClassicalClustering, AdvancedClustering, N2DClustering, DBScanClustering
from iss.tools import Tools


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

    clustering, clustering_config = Tools.load_clustering(CONFIG, clustering_type, clustering_version, clustering_model_type, clustering_model_name)

    if drop:
        db_manager.drop_embedding_partition(clustering_type, clustering_version, clustering_model_type, clustering_model_name)

    if clustering_type == 'n2d':
        clustering = N2DClustering(clustering_config)
    elif clustering_type == 'classical':
        clustering = ClassicalClustering(clustering_config)
    elif clustering_type == 'dbscan':
        clustering = DBScanClustering(clustering_config)
    else:
        raise Exception

    clustering.load()
    model, model_config = Tools.load_model(CONFIG, clustering_model_type, clustering_model_name)
    filenames = Tools.list_directory_filenames(CONFIG.get('directory')['collections'])
    generator = Tools.load_latent_representation(CONFIG, model, model_config, filenames, 496, None, True)

    count = 0
    for ids, latents in generator:
        pictures_embedding = clustering.predict_embedding(latents)
        pictures_label = clustering.predict_label(pictures_embedding)
        rows = []
        for i, id in enumerate(ids):
            rows.append((
                id,
                float(np.round(pictures_embedding[i][0], 4)),
                float(np.round(pictures_embedding[i][1], 4)),
                int(pictures_label[i]),
                clustering_type,
                clustering_version,
                clustering_model_type,
                clustering_model_name
            ))
        count += db_manager.insert_row_pictures_embedding(rows)
        print("Nombre d'insertion: %s / %s" % (count, len(filenames)))


    return


def main(action = 'populate_embedding'):

    db_manager = Tools.create_db_manager(CONFIG)

    if action == 'population_locations':
        populate_locations(CONFIG, db_manager)
    elif action == 'populate_embedding':
        db_manager.create_pictures_embedding_table(False)
        to_load = [
            {'clustering_type': 'n2d', 'clustering_version': 1, 'clustering_model_type': 'simple_conv', 'clustering_model_name': 'model_colab', 'drop': False},
            {'clustering_type': 'n2d', 'clustering_version': 2, 'clustering_model_type': 'simple_conv', 'clustering_model_name': 'model_colab', 'drop': False},
            {'clustering_type': 'n2d', 'clustering_version': 3, 'clustering_model_type': 'simple_conv', 'clustering_model_name': 'model_colab', 'drop': False},
            ]
        for kwargs in to_load:
            try:
                populate_embedding(CONFIG, db_manager, **kwargs)
            except Exception as err:
                print(err)
                pass
            
    else:
        pass


if __name__ == '__main__':
    main()