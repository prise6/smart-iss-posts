# -*- coding: utf-8 -*-
import os
from iss.tools import Tools

class AbstractClustering:

    def __init__(self, config, pictures_id, pictures_np):

        self.config = config
        self.save_directory = os.path.join(self.config['save_directory'], '%s_%s_%s' % (self.config['model']['type'], self.config['model']['name'], self.config['version']))
        self.pictures_id = pictures_id
        self.pictures_np = pictures_np
        self.final_labels = None
        self.colors = None

        Tools.create_dir_if_not_exists(self.save_directory)

    def compute_final_labels(self):
        raise NotImplementedError

    def get_results(self):
        return list(zip(self.pictures_id, self.final_labels, self.pictures_np))
        
    def compute_silhouette_score(self):
        raise NotImplementedError

    def compute_colors(self):
        n_classes = len(list(set(self.final_labels)))
        self.colors = [Tools.get_color_from_label(label, n_classes) for label in self.final_labels]
        return self

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError   