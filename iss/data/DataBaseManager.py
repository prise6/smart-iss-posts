# -*- coding: utf-8 -*-


class DataBaseManager:

	def __init__(self, connexion, config):
		self.conn = connexion
		self.config = config
		self.cursor = self.conn.cursor()


	def createPicturesTable(self, force = False):

		if force:
			self.cursor.execute("DROP TABLE IF EXISTS `iss`.`pictures`;")

		self.cursor.execute("""
CREATE TABLE `iss`.`pictures` (
  `pictures_latitude` FLOAT(10, 6) NULL,
  `pictures_longitude` FLOAT(10, 6 ) NULL ,
  `pictures_id` VARCHAR( 15 ) PRIMARY KEY ,
  `pictures_timestamp` TIMESTAMP NULL ,
  `pictures_location` TEXT NULL
) ENGINE = MYISAM ;
			""")


	def insertRowPictures(self, array):

		sql_insert_template = "INSERT INTO `iss`.`pictures` (pictures_latitude, pictures_longitude, pictures_id, pictures_timestamp, pictures_location) VALUES (%s, %s, %s, %s, %s);"

		self.cursor.executemany(sql_insert_template, array)
		self.conn.commit()

		return self.cursor.rowcount

	def select(self, array):
		sql = """
SELECT tmp.* FROM (
	SELECT 
	COUNT(*) as nb,
	pictures_location as location

	FROM iss.pictures
	GROUP BY pictures_location
	) as tmp 
ORDER BY nb DESC
		"""