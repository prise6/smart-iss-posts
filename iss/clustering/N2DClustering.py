# -*- coding: utf-8 -*-

import os
import numpy as np
import umap
from iss.tools import Tools
from iss.clustering import AbstractClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.externals import joblib

class N2DClustering(AbstractClustering):
    """
    Cf: https://github.com/rymc/n2d
    """

    def __init__(self, config, pictures_id = None, pictures_np = None):

        super().__init__(config, pictures_id, pictures_np)
        
        self.umap_args = self.config['umap']
        self.umap_fit = None
        self.umap_embedding = None
        self.umap_save_name = 'UMAP_model.pkl'

        self.kmeans_fit = None
        self.kmeans_args = self.config['kmeans']
        self.kmeans_labels = None
        self.kmeans_centers = []
        self.kmeans_save_name = "kmeans_model.pkl"

        
    def compute_umap(self):
        self.umap_fit = umap.UMAP(**self.umap_args)
        self.umap_embedding = self.umap_fit.fit_transform(self.pictures_np)
        return self

    def compute_kmeans(self):
        self.kmeans_fit = KMeans(**self.kmeans_args)
        self.kmeans_fit.fit(self.umap_embedding)
        self.kmeans_labels = self.kmeans_fit.labels_
        return self

    def compute_final_labels(self):
        self.final_labels  = self.kmeans_labels
        return self

    def compute_silhouette_score(self):
        self.silhouette_score = silhouette_samples(self.pictures_np, self.final_labels)
        self.silhouette_score_labels = {cluster: np.mean(self.silhouette_score[self.final_labels == cluster]) for 
        cluster in np.unique(self.final_labels)}
        return self.silhouette_score_labels

    def save(self):
        Tools.create_dir_if_not_exists(self.save_directory)

        joblib.dump(self.umap_fit, os.path.join(self.save_directory, self.umap_save_name))
        joblib.dump(self.kmeans_fit, os.path.join(self.save_directory, self.kmeans_save_name))

    def load(self):
        self.umap_fit = joblib.load(os.path.join(self.save_directory, self.pca_save_name))
        self.kmeans_fit = joblib.load(os.path.join(self.save_directory, self.kmeans_save_name))