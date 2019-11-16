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
from iss.models import SimpleConvAutoEncoder
from iss.clustering import ClassicalClustering, AdvancedClustering, N2DClustering

## variable globales

_MODEL_TYPE = 'simple_conv'
_MODEL_NAME = 'model_colab'
_BATCH_SIZE = 496
_N_BATCH = 10
_DEBUG = True
_CLUSTERING_TYPE = 'n2d'
_OUTPUT_IMAGE_WIDTH = 96
_OUTPUT_IMAGE_HEIGHT = 54
_MOSAIC_NROW = 10
_MOSAIC_NCOL_MAX = 10


## Charger le modèle
CONFIG.get('models')[_MODEL_TYPE]['model_name'] = _MODEL_NAME
model = SimpleConvAutoEncoder(CONFIG.get('models')[_MODEL_TYPE])
model_config = CONFIG.get('models')[_MODEL_TYPE]

## Charger les images
filenames = Tools.list_directory_filenames(os.path.join(CONFIG.get('directory')['autoencoder']['train']))
generator_imgs = Tools.generator_np_picture_from_filenames(filenames, target_size = (model_config['input_height'], model_config['input_width']), batch = _BATCH_SIZE, nb_batch = _N_BATCH)

pictures_id, pictures_preds = Tools.encoded_pictures_from_generator(generator_imgs, model)
intermediate_output = pictures_preds.reshape((pictures_preds.shape[0], model_config['latent_width']*model_config['latent_height']*model_config['latent_channel']))


if _DEBUG:
    for i, p_id in enumerate(pictures_id[:2]):
        print("%s: %s" % (p_id, pictures_preds[i]))
    print(len(pictures_id))
    print(len(intermediate_output))


## Clustering
if _CLUSTERING_TYPE == 'classical':
    if _DEBUG:
        print("Classical Clustering")
    clustering = ClassicalClustering(CONFIG.get('clustering')['classical'], pictures_id, intermediate_output)
    clustering.compute_pca()
    clustering.compute_kmeans()
    clustering.compute_kmeans_centers()
    clustering.compute_cah()
    clustering.compute_final_labels()
    clustering.compute_tsne()
    clustering.compute_colors()
elif _CLUSTERING_TYPE == 'advanced':
    if _DEBUG:
        print("Advanced Clustering")
    clustering = AdvancedClustering(CONFIG.get('clustering')['classical'], pictures_id, intermediate_output)
elif _CLUSTERING_TYPE == 'n2d':
    if _DEBUG:
        print("Not2Deep Clustering")
    clustering = N2DClustering(CONFIG.get('clustering')['n2d'], pictures_id, intermediate_output)
    clustering.compute_umap()
    clustering.compute_kmeans()
    clustering.compute_final_labels()
    clustering.compute_colors()

silhouettes = clustering.compute_silhouette_score()
clustering_res = clustering.get_results()

if _DEBUG:
    print(clustering_res[:2])
    print(silhouettes)


if _CLUSTERING_TYPE in ['classical']:
    ## Graphs of PCA and final clusters
    fig, ax = plt.subplots(figsize=(24, 14))
    scatter = ax.scatter(clustering.pca_reduction[:, 0], clustering.pca_reduction[:, 1], c = clustering.colors)
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.savefig(os.path.join(CONFIG.get('clustering')[_CLUSTERING_TYPE]['save_directory'], 'pca_clusters.png'))

if _CLUSTERING_TYPE in ['classical']:
    ## Graphs of TSNE and final clusters
    fig, ax = plt.subplots(figsize=(24, 14))
    classes = clustering.final_labels
    scatter = ax.scatter(clustering.tsne_embedding[:, 0], clustering.tsne_embedding[:, 1], c = clustering.colors)
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.savefig(os.path.join(CONFIG.get('clustering')[_CLUSTERING_TYPE]['save_directory'], 'tsne_clusters.png'))

if _CLUSTERING_TYPE in ['n2d']:
    ## Graphs of TSNE and final clusters
    fig, ax = plt.subplots(figsize=(24, 14))
    classes = clustering.final_labels
    scatter = ax.scatter(clustering.umap_embedding[:, 0], clustering.umap_embedding[:, 1], c = clustering.colors)
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.savefig(os.path.join(CONFIG.get('clustering')[_CLUSTERING_TYPE]['save_directory'], 'umap_clusters.png'))

if _CLUSTERING_TYPE in ['n2d']:
    filenames = [os.path.join(CONFIG.get('directory')['collections'], "%s.jpg" % one_res[0]) for one_res in clustering_res]
    images_array = [Tools.read_np_picture(img_filename, target_size = (54, 96)) for img_filename in filenames]
    base64_images = [Tools.base64_image(img) for img in images_array]

    print(clustering.umap_embedding)
    print(clustering.umap_embedding.shape)

    x = clustering.umap_embedding[:, 0]
    y = clustering.umap_embedding[:, 1]

    df = pd.DataFrame({'x': x, 'y': y})
    df['image'] = base64_images
    df['label'] = clustering.final_labels.astype(str)
    df['color'] = df['label'].apply(Tools.get_color_from_label)

    datasource = ColumnDataSource(df)

    output_file(os.path.join(CONFIG.get('clustering')[_CLUSTERING_TYPE]['save_directory'], 'umap_bokeh.html'))

    plot_figure = figure(
        title='UMAP projection of iss clusters',
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


if _CLUSTERING_TYPE in ['classical']:
    ## Dendogram
    fig, ax = plt.subplots(figsize=(24, 14))
    plt.title('Hierarchical Clustering Dendrogram')
    Tools.plot_dendrogram(clustering.cah_fit, labels=clustering.cah_labels)
    plt.savefig(os.path.join(CONFIG.get('clustering')[_CLUSTERING_TYPE]['save_directory'], 'dendograms.png'))


## Silhouette
fig, ax = plt.subplots(figsize=(12, 7))
ax.bar(silhouettes.keys(), silhouettes.values(), align='center')
ax.set_xticks(list(silhouettes.keys()))
ax.set_xticklabels(list(silhouettes.keys()))
plt.savefig(os.path.join(CONFIG.get('clustering')[_CLUSTERING_TYPE]['save_directory'], 'silhouettes_score.png'))


## Mosaic of each cluster
clusters_id = np.unique(clustering.final_labels)
for cluster_id in clusters_id:
    cluster_image_filenames = [os.path.join(CONFIG.get('directory')['collections'], "%s.jpg" % one_res[0]) for one_res in clustering_res if one_res[1] == cluster_id]

    images_array = [Tools.read_np_picture(img_filename, target_size = (_OUTPUT_IMAGE_HEIGHT, _OUTPUT_IMAGE_WIDTH)) for img_filename in cluster_image_filenames]

    img = Tools.display_mosaic(images_array, nrow = _MOSAIC_NROW, ncol_max = _MOSAIC_NCOL_MAX)
    img.save(os.path.join(CONFIG.get('clustering')[_CLUSTERING_TYPE]['save_directory'], "cluster_%s.png" % str(cluster_id).zfill(2)), "PNG")
