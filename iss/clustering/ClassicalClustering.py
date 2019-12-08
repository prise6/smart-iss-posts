# -*- coding: utf-8 -*-

import os
import numpy as np
from iss.clustering import AbstractClustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples
from iss.tools import Tools
from sklearn.externals import joblib
from sklearn.manifold import TSNE

class ClassicalClustering(AbstractClustering):

	def __init__(self, config, pictures_id = None, pictures_np = None):

		super().__init__(config, pictures_id, pictures_np)

		self.pca_fit = None
		self.pca_args = self.config['PCA']
		self.pca_reduction = None
		self.pca_save_name = "PCA_model.pkl"

		self.kmeans_fit = None
		self.kmeans_args = self.config['kmeans']
		self.kmeans_labels = None
		self.kmeans_centers = []
		self.kmeans_save_name = "kmeans_model.pkl"


		self.cah_fit = None
		self.cah_args = self.config['CAH']
		self.cah_labels = None
		self.cah_save_name = "cah_model.pkl"
		
		self.tsne_fit = None
		self.tsne_args = self.config['TSNE']
		self.tsne_embedding = None

		self.final_labels = None
		self.silhouette_score_labels = {}


	def compute_pca(self):

		self.pca_fit = PCA(**self.pca_args)
		self.pca_fit.fit(self.pictures_np)
		self.pca_reduction = self.pca_fit.transform(self.pictures_np)

		return self

	def compute_kmeans(self):
		self.kmeans_fit = KMeans(**self.kmeans_args)
		self.kmeans_fit.fit(self.pca_reduction)
		self.kmeans_labels = self.kmeans_fit.labels_
		return self

	def compute_kmeans_centers(self):
		for cl in range(self.kmeans_args['n_clusters']):
			tmp = self.pca_reduction[np.where(self.kmeans_labels == cl)]
			self.kmeans_centers.append(np.mean(tmp, axis = 0))
		return self

	def compute_cah(self):

		self.cah_fit = AgglomerativeClustering(**self.cah_args)
		self.cah_fit.fit_predict(self.kmeans_centers)
		self.cah_labels = self.cah_fit.labels_
		return self

	def compute_final_labels(self):
		self.final_labels = np.array([self.cah_labels[old_cl] for old_cl in self.kmeans_labels])

	def compute_tsne(self):
		self.tsne_fit = TSNE(**self.tsne_args)
		self.tsne_embedding = self.tsne_fit.fit_transform(self.pca_reduction)
		return self
		
	def get_results(self):
		return list(zip(self.pictures_id, self.final_labels, self.kmeans_labels, self.pictures_np))

	def compute_silhouette_score(self):
		self.silhouette_score = silhouette_samples(self.pictures_np, self.final_labels)
		self.silhouette_score_labels = {cluster: np.mean(self.silhouette_score[self.final_labels == cluster]) for 
		cluster in np.unique(self.final_labels)}
		return self.silhouette_score_labels


	def save(self):
		Tools.create_dir_if_not_exists(self.save_directory)

		joblib.dump(self.pca_fit, os.path.join(self.save_directory, self.pca_save_name))
		joblib.dump(self.kmeans_fit, os.path.join(self.save_directory, self.kmeans_save_name))
		joblib.dump(self.cah_fit, os.path.join(self.save_directory, self.cah_save_name))

	def load(self):
		self.pca_fit = joblib.load(os.path.join(self.save_directory, self.pca_save_name))
		self.kmeans_fit = joblib.load(os.path.join(self.save_directory, self.kmeans_save_name))
		self.cah_fit = joblib.load(os.path.join(self.save_directory, self.cah_save_name))


