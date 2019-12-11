# -*- coding: utf-8 -*-


class MysqlDataBaseManager:

	def __init__(self, connexion, config):
		self.conn = connexion
		self.config = config
		self.cursor = self.conn.cursor()


	def create_pictures_location_table(self, force = False):

		if force:
			self.cursor.execute("DROP TABLE IF EXISTS `iss`.`pictures_location`;")

		self.cursor.execute("""
CREATE TABLE IF NOT EXISTS `iss`.`pictures_location` (
  `pictures_latitude` FLOAT(10, 6) NULL,
  `pictures_longitude` FLOAT(10, 6 ) NULL ,
  `pictures_id` VARCHAR( 15 ) PRIMARY KEY ,
  `pictures_timestamp` TIMESTAMP NULL ,
  `pictures_location_text` TEXT NULL
) ENGINE = MYISAM ;
			""")


	def insert_row_pictures_location(self, array):

		sql_insert_template = "INSERT INTO `iss`.`pictures_location` (pictures_latitude, pictures_longitude, pictures_id, pictures_timestamp, pictures_location_text) VALUES (%s, %s, %s, %s, %s);"

		self.cursor.executemany(sql_insert_template, array)
		self.conn.commit()

		return self.cursor.rowcount


	def create_pictures_embedding_table(self, force = False):

		if force:
			self.cursor.execute("DROP TABLE IF EXISTS `iss`.`pictures_embedding`;")

		self.cursor.execute("""
CREATE TABLE IF NOT EXISTS `iss`.`pictures_embedding` (
  `pictures_id` VARCHAR( 15 ) ,
  `pictures_x` FLOAT(8, 4),
  `pictures_y` FLOAT(8, 4),
  `clustering_type` VARCHAR(15),
  `clustering_version` VARCHAR(5),
  `clustering_model_type` VARCHAR(15),
  `clustering_model_name` VARCHAR(15),
  UNIQUE KEY `unique_key` (`pictures_id`,`clustering_type`, `clustering_version`, `clustering_model_type`,`clustering_model_name`),
  KEY `index_key_1` (`pictures_id`)
) ENGINE = MYISAM ;
			""")


	def drop_embedding_partition(self, clustering_type, clustering_version, clustering_model_type, clustering_model_name):

		req = "DELETE FROM `iss`.`pictures_embedding` WHERE clustering_type = %s AND clustering_version = %s AND clustering_model_type = %s AND clustering_model_name = %s"

		self.cursor.execute(req, (clustering_type, clustering_version, clustering_model_type, clustering_model_name))

		self.conn.commit()

		return self.cursor.rowcount


	def insert_row_pictures_embedding(self, array):

		sql_insert_template = "INSERT INTO `iss`.`pictures_embedding` (pictures_id, pictures_x, pictures_y, clustering_type, clustering_version, clustering_model_type, clustering_model_name) VALUES (%s, %s, %s, %s, %s, %s, %s);"

		self.cursor.executemany(sql_insert_template, array)
		self.conn.commit()

		return self.cursor.rowcount
