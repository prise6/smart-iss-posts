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

        self.kmeans_fit = None
        self.kmeans_args = self.config['kmeans']
        self.kmeans_labels = None
        self.kmeans_centers = []
        self.kmeans_save_name = "kmeans_model_v%s.pkl" % (self.config['version'])

        
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

