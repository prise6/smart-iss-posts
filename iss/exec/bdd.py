import os
import time
import mysql.connector
import pandas as pd
import datetime as dt

from iss.init_config import CONFIG
from iss.data.DataBaseManager import DataBaseManager

CON_MYSQL = mysql.connector.connect(
    host = CONFIG.get('mysql')['database']['server'],
    user = CONFIG.get('mysql')['database']['user'],
    passwd = CONFIG.get('mysql')['database']['password'],
    database = CONFIG.get('mysql')['database']['name'],
    port = CONFIG.get('mysql')['database']['port']
)

dbm = DataBaseManager(CON_MYSQL, CONFIG)


history = pd.read_csv(os.path.join(CONFIG.get("directory")['data_dir'], "raw", "history", "history.txt"), sep=";", names=['latitude', 'longitude', 'id', 'location'])
history['timestamp'] = pd.to_datetime(history.id, format="%Y%m%d-%H%M%S").dt.strftime("%Y-%m-%d %H:%M:%S")
history.fillna('NULL', inplace=True)
history = history[['latitude', 'longitude', 'id', 'timestamp', 'location']]
history_tuple = [tuple(x) for x in history.values]

dbm.createPicturesTable(force=True)
count = dbm.insertRowPictures(history_tuple)

print(count)