# -*- coding: utf-8 -*-

import os
import numpy as np
import umap
import hdbscan
from iss.tools import Tools
from iss.clustering import AbstractClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.externals import joblib

class DBScanClustering(AbstractClustering):
    """
    Cf: https://umap-learn.readthedocs.io/en/latest/clustering.html
    """

    def __init__(self, config, pictures_id = None, pictures_np = None):

        super().__init__(config, pictures_id, pictures_np)
        
        self.umap_args = self.config['umap']
        self.umap_fit = None
        self.umap_embedding = None
        self.umap_save_name = 'UMAP_model.pkl'

        self.dbscan_fit = None
        self.dbscan_args = self.config['dbscan']
        self.dbscan_labels = None
        self.dbscan_centers = []
        self.dbscan_save_name = "dbscan_model.pkl"

        
    def compute_umap(self):
        self.umap_fit = umap.UMAP(**self.umap_args)
        self.umap_embedding = self.umap_fit.fit_transform(self.pictures_np)
        return self

    def compute_dbscan(self):
        self.dbscan_fit = hdbscan.HDBSCAN(**self.dbscan_args)
        self.dbscan_fit.fit(self.umap_embedding)
        self.dbscan_labels = self.dbscan_fit.labels_
        return self

    def compute_final_labels(self):
        self.final_labels  = self.dbscan_labels
        return self

    def compute_silhouette_score(self):
        self.silhouette_score = silhouette_samples(self.pictures_np, self.final_labels)
        self.silhouette_score_labels = {cluster: np.mean(self.silhouette_score[self.final_labels == cluster]) for 
        cluster in np.unique(self.final_labels)}
        return self.silhouette_score_labels

    def predict_embedding(self, pictures_np):
        return self.umap_fit.transform(pictures_np)

    def save(self):
        Tools.create_dir_if_not_exists(self.save_directory)

        joblib.dump(self.umap_fit, os.path.join(self.save_directory, self.umap_save_name))
        joblib.dump(self.dbscan_fit, os.path.join(self.save_directory, self.dbscan_save_name))

    def load(self):
        self.umap_fit = joblib.load(os.path.join(self.save_directory, self.umap_save_name))
        self.dbscan_fit = joblib.load(os.path.join(self.save_directory, self.dbscan_save_name))