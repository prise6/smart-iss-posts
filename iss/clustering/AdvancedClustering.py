# -*- coding: utf-8 -*-

import os
import numpy as np
from iss.clustering import AbstractClustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from iss.tools import Tools
from sklearn.externals import joblib
import pandas as pd

class AdvancedClustering(AbstractClustering):

    def __init__(self, config, pictures_id = None, pictures_np = None):

        super().__init__(config, pictures_id, pictures_np)

        self.pca_fit = None
        self.pca_args = self.config['PCA']
        self.pca_reduction = None
        self.pca_save_name = "PCA_model_v%s.pkl" % (self.config['version'])

        self.kmeans_fit = None
        self.kmeans_args = self.config['strong_kmeans']
        self.kmeans_labels = None
        self.kmeans_centers = []
        self.kmeans_save_name = "kmeans_model_v%s.pkl" % (self.config['version'])

        self.dbscan_fit = None
        self.dbscan_args = self.config['dbscan']
        self.dbscan_labels = None
        self.dbscan_save_name = "dbscan_model_v%s.pkl" % (self.config['version'])

        self.final_labels = None


    def compute_pca(self):

        np.random.seed(self.pca_args['random_state'])
        self.pca_fit = PCA(**self.pca_args)
        self.pca_fit.fit(self.pictures_np)
        self.pca_reduction = self.pca_fit.transform(self.pictures_np)
        print(self.pca_reduction)
        return self

    def compute_kmeans(self):

        tmp_labels = pd.DataFrame()
        tmp_iter = self.kmeans_args['iter']
        tmp_range = range(0, tmp_iter)
        tmp_low = self.kmeans_args['low']
        tmp_high = self.kmeans_args['high']
        tmp_treshold = self.kmeans_args['threshold']
        tmp_cols = ['run_%s' % i for i in tmp_range]
        np.random.seed(self.kmeans_args['seed']*2)
        tmp_n_clusters = np.random.randint(low = tmp_low, high = tmp_high, size = tmp_iter)
        print(tmp_n_clusters)

        for i in tmp_range:
            km_model = KMeans(n_clusters = tmp_n_clusters[i], random_state = self.kmeans_args['seed']+i)
            km_res = km_model.fit(self.pca_reduction)
            tmp_labels[tmp_cols[i]] = km_res.labels_

        tmp_labels['dummy'] = 1
        tmp_labels['group_id'] = tmp_labels.groupby(by = tmp_cols, as_index = False).grouper.group_info[0]
        tmp_labels['count'] = tmp_labels.groupby(by = 'group_id', as_index = False)['dummy'].transform(np.size)
        tmp_labels = tmp_labels.drop(labels = 'dummy', axis = 1)

        tmp_group_id = tmp_labels[tmp_labels['count'] >= tmp_treshold]['group_id'].unique()

        print(tmp_group_id)

        pca_init = np.zeros((len(tmp_group_id), self.pca_reduction.shape[1]))

        for i in range(0, len(tmp_group_id)):
            gp_id = tmp_group_id[i]
            index_sel = tmp_labels[tmp_labels['group_id'] == gp_id].index
            pca_init[i, :] = np.mean(self.pca_reduction[index_sel, :], axis = 0)


        self.kmeans_fit = KMeans(n_clusters = pca_init.shape[0], init = pca_init, n_init = 1, random_state = self.kmeans_args['seed']+tmp_iter)
        self.kmeans_fit.fit(self.pca_reduction)
        self.kmeans_labels = self.kmeans_fit.labels_
        return self

    def compute_kmeans_centers(self):
        for cl in list(np.unique(self.kmeans_fit.labels_)):
            tmp = self.pca_reduction[np.where(self.kmeans_labels == cl)]
            self.kmeans_centers.append(np.mean(tmp, axis = 0))
        return self
    
    def compute_dbscan(self):
        self.dbscan_fit = DBSCAN(**self.dbscan_args)
        self.dbscan_fit.fit_predict(self.kmeans_centers)
        self.dbscan_labels = self.dbscan_fit.labels_
        return self

    def compute_dbscan_labels(self):
        self.final_labels = [self.dbscan_labels[old_cl] for old_cl in self.kmeans_labels]

    def get_zip_results(self):
        return zip(self.pictures_id, self.final_labels, self.kmeans_labels, self.pictures_np)

    def save(self):
        Tools.create_dir_if_not_exists(self.config['save_directory'])

        joblib.dump(self.pca_fit, os.path.join(self.config['save_directory'], self.pca_save_name))
        joblib.dump(self.kmeans_fit, os.path.join(self.config['save_directory'], self.kmeans_save_name))
        joblib.dump(self.dbscan_fit, os.path.join(self.config['save_directory'], self.dbscan_save_name))

    def load(self):
        self.pca_fit = joblib.load(os.path.join(self.config['save_directory'], self.pca_save_name))
        self.kmeans_fit = joblib.load(os.path.join(self.config['save_directory'], self.kmeans_save_name))
        self.dbscan_fit = joblib.load(os.path.join(self.config['save_directory'], self.dbscan_save_name))

