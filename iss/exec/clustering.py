import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper

from iss.init_config import CONFIG
from iss.tools import Tools
from iss.models import SimpleConvAutoEncoder, SimpleAutoEncoder
from iss.clustering import ClassicalClustering, AdvancedClustering, N2DClustering


_DEBUG = True


def run_clustering(config, clustering_type, pictures_id, intermediate_output):
    """
    Apply clustering on images
    """

    if clustering_type == 'classical':
        if _DEBUG:
            print("Classical Clustering")
        clustering = ClassicalClustering(config.get('clustering')['classical'], pictures_id, intermediate_output)
        clustering.compute_pca()
        clustering.compute_kmeans()
        clustering.compute_kmeans_centers()
        clustering.compute_cah()
        clustering.compute_final_labels()
        clustering.compute_tsne()
        clustering.compute_colors()
    elif clustering_type == 'advanced':
        if _DEBUG:
            print("Advanced Clustering")
        clustering = AdvancedClustering(config.get('clustering')['classical'], pictures_id, intermediate_output)
    elif clustering_type == 'n2d':
        if _DEBUG:
            print("Not2Deep Clustering")
        clustering = N2DClustering(config.get('clustering')['n2d'], pictures_id, intermediate_output)
        clustering.compute_umap()
        clustering.compute_kmeans()
        clustering.compute_final_labels()
        clustering.compute_colors()

    return clustering


def run_plots(config, clustering_type, clustering):
    """
    Plots specifics graphs
    """

    if clustering_type in ['classical']:
        ## Graphs of PCA and final clusters
        fig, ax = plt.subplots(figsize=(24, 14))
        scatter = ax.scatter(clustering.pca_reduction[:, 0], clustering.pca_reduction[:, 1], c = clustering.colors)
        legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
        ax.add_artist(legend1)
        plt.savefig(os.path.join(clustering.save_directory, 'pca_clusters.png'))

    if clustering_type in ['classical']:
        ## Graphs of TSNE and final clusters
        fig, ax = plt.subplots(figsize=(24, 14))
        classes = clustering.final_labels
        scatter = ax.scatter(clustering.tsne_embedding[:, 0], clustering.tsne_embedding[:, 1], c = clustering.colors)
        legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
        ax.add_artist(legend1)
        plt.savefig(os.path.join(clustering.save_directory, 'tsne_clusters.png'))

    if clustering_type in ['n2d']:
        ## Graphs of TSNE and final clusters
        fig, ax = plt.subplots(figsize=(24, 14))
        classes = clustering.final_labels
        scatter = ax.scatter(clustering.umap_embedding[:, 0], clustering.umap_embedding[:, 1], c = clustering.colors)
        legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
        ax.add_artist(legend1)
        plt.savefig(os.path.join(clustering.save_directory, 'umap_clusters.png'))

    if clustering_type in ['n2d', 'classical']:
        filenames = [os.path.join(config.get('directory')['collections'], "%s.jpg" % one_res[0]) for one_res in clustering.get_results()]
        images_array = [Tools.read_np_picture(img_filename, target_size = (54, 96)) for img_filename in filenames]
        base64_images = [Tools.base64_image(img) for img in images_array]

        if clustering_type == 'n2d':
            x = clustering.umap_embedding[:, 0]
            y = clustering.umap_embedding[:, 1]
            html_file = 'umap_bokeh.html'
            title = 'UMAP projection of iss clusters'
        elif clustering_type == 'classical':
            x = clustering.tsne_embedding[:, 0]
            y = clustering.tsne_embedding[:, 1]
            html_file = 'tsne_bokeh.html'
            title = 't-SNE projection of iss clusters'

        df = pd.DataFrame({'x': x, 'y': y})
        df['image'] = base64_images
        df['label'] = clustering.final_labels.astype(str)
        df['color'] = df['label'].apply(Tools.get_color_from_label)

        datasource = ColumnDataSource(df)

        output_file(os.path.join(clustering.save_directory, html_file))

        plot_figure = figure(
            title=title,
            # plot_width=1200,
            # plot_height=1200,
            tools=('pan, wheel_zoom, reset')
        )

        plot_figure.add_tools(HoverTool(tooltips="""
        <div>
            <div>
                <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
            </div>
            <div>
                <span style='font-size: 16px'>Cluster:</span>
                <span style='font-size: 18px'>@label</span>
            </div>
        </div>
        """))


        plot_figure.circle(
            'x',
            'y',
            source=datasource,
            color=dict(field='color'),
            line_alpha=0.6,
            fill_alpha=0.6,
            size=4
        )

        show(plot_figure)


    if clustering_type in ['classical']:
        ## Dendogram
        fig, ax = plt.subplots(figsize=(24, 14))
        plt.title('Hierarchical Clustering Dendrogram')
        Tools.plot_dendrogram(clustering.cah_fit, labels=clustering.cah_labels)
        plt.savefig(os.path.join(clustering.save_directory, 'dendograms.png'))

    return True

def plot_silhouette(config, clustering_type, clustering):
    
    silhouettes = clustering.compute_silhouette_score()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(silhouettes.keys(), silhouettes.values(), align='center')
    ax.set_xticks(list(silhouettes.keys()))
    ax.set_xticklabels(list(silhouettes.keys()))
    plt.savefig(os.path.join(clustering.save_directory, 'silhouettes_score.png'))

    return silhouettes


def plot_mosaics(config, clustering_type, clustering, output_image_width, output_image_height, mosaic_nrow, mosaic_ncol_max):
    """
    Mosaic of each cluster
    """
    clusters_id = np.unique(clustering.final_labels)
    clustering_res = clustering.get_results()

    for cluster_id in clusters_id:
        cluster_image_filenames = [os.path.join(config.get('directory')['collections'], "%s.jpg" % one_res[0]) for one_res in clustering_res if one_res[1] == cluster_id]

        images_array = [Tools.read_np_picture(img_filename, target_size = (output_image_height, output_image_width)) for img_filename in cluster_image_filenames]

        img = Tools.display_mosaic(images_array, nrow = mosaic_nrow, ncol_max = mosaic_ncol_max)
        img.save(os.path.join(clustering.save_directory, "cluster_%s.png" % str(cluster_id).zfill(2)), "PNG")

    return clusters_id


def main():
    _CLUSTERING_TYPE = 'n2d'
    _BATCH_SIZE = 496
    _N_BATCH = 10
    _PLOTS = True
    _MOSAICS = True
    _SILHOUETTE = True
    _OUTPUT_IMAGE_WIDTH = 96
    _OUTPUT_IMAGE_HEIGHT = 54
    _MOSAIC_NROW = 10
    _MOSAIC_NCOL_MAX = 10

    model, model_config = Tools.load_model(CONFIG, CONFIG.get('clustering')[_CLUSTERING_TYPE]['model']['type'], CONFIG.get('clustering')[_CLUSTERING_TYPE]['model']['name'])
    filenames = Tools.list_directory_filenames(CONFIG.get('sampling')['autoencoder']['directory']['train'])
    pictures_id, intermediate_output = Tools.load_latent_representation(CONFIG, model, model_config, filenames, _BATCH_SIZE, _N_BATCH, False)
            
    clustering = run_clustering(CONFIG, _CLUSTERING_TYPE, pictures_id, intermediate_output)
    
    clustering.save()

    if _PLOTS:
        run_plots(CONFIG, _CLUSTERING_TYPE, clustering)

    if _SILHOUETTE:
        plot_silhouette(CONFIG, _CLUSTERING_TYPE, clustering)

    if _MOSAICS:
        plot_mosaics(CONFIG, _CLUSTERING_TYPE, clustering, _OUTPUT_IMAGE_WIDTH, _OUTPUT_IMAGE_HEIGHT, _MOSAIC_NROW, _MOSAIC_NCOL_MAX)


if __name__ == '__main__':
    main()
