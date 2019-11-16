#%% [markdown]
# # Clustering classique

#%% [markdown]
# ## import classique
import os

#%%
%load_ext autoreload
%autoreload 2
os.chdir('/home/jovyan/work')

#%% [markdown]
# ## Import iss

#%%
from iss.tools import Config
from iss.tools import Tools
from iss.models import SimpleConvAutoEncoder
from iss.clustering import ClassicalClustering
from iss.clustering import AdvancedClustering
from dotenv import find_dotenv, load_dotenv
import numpy as np

#%% [markdown]
# ## Chargement de la config

#%%
load_dotenv(find_dotenv())
cfg = Config(project_dir = os.getenv("PROJECT_DIR"), mode = os.getenv("MODE"))

#%% [markdown]
# ## Chargement du modèle

#%%
## charger le modèle

model_type = 'simple_conv'
cfg.get('models')[model_type]['model_name'] = 'model_colab'
model = SimpleConvAutoEncoder(cfg.get('models')[model_type])

#%% [markdown]
## Chargement des images

#%%
filenames = Tools.list_directory_filenames('data/processed/models/autoencoder/train/k/')
generator_imgs = Tools.generator_np_picture_from_filenames(filenames, target_size = (27, 48), batch = 496, nb_batch = 10, scale = 1/255)


#%%
pictures_id, pictures_preds = Tools.encoded_pictures_from_generator(generator_imgs, model)

#%%
intermediate_output = pictures_preds.reshape((pictures_preds.shape[0], 3*6*16))


#%%
clustering = AdvancedClustering(cfg.get('clustering')['advanced'], pictures_id, intermediate_output)


#%%
clustering.compute_pca()


#%%
clustering.compute_kmeans()

#%%
clustering.compute_kmeans_centers()

#%%
len(clustering.kmeans_centers)

#%%
clustering.dbscan_args = {'eps': 50, 'min_samples':1}
clustering.compute_dbscan()

#%%
clustering.compute_dbscan_labels()

#%%
len(clustering.final_labels)

#%%
np.unique(clustering.final_labels, return_counts = True)

#%%[markdown]
# # Graphiques

#%%
def select_cluster(clustering, id_cluster):
    return [os.path.join('data/processed/models/autoencoder/train/k/', res[0] + '.jpg') for res in clustering.get_zip_results() if res[2] == id_cluster]


#%%
for cl in np.unique(clustering.kmeans_labels):
    print("Cluster %s" % (cl))
    res_tmp = select_cluster(clustering, cl)
    if len(res_tmp) >= 0:
        print(len(res_tmp))
        image_array = [Tools.read_np_picture(f, target_size = (54, 96)) for f in res_tmp[:100]]
        img = Tools.display_mosaic(image_array, nrow = 10)
        fig = plt.figure(1, figsize=(12, 7))
        plt.imshow(img, aspect = 'auto')
        plt.show()

#%% [markdown]
# ## faut essayer de faire des paquets

#%%
from sklearn.manifold import TSNE

output_tnse = TSNE(n_components=2).fit_transform(clustering.pca_reduction)


#%%
plt.scatter(
    output_tnse[:,0],
    output_tnse[:,1],
    c = clustering.kmeans_labels
)
plt.show()

#%%
from sklearn.cluster import KMeans

tmp_km = KMeans(n_clusters = 15)
tmp_res = tmp_km.fit(output_tnse)

#%%
tmp_res.labels_

#%%
plt.scatter(
    output_tnse[:,0],
    output_tnse[:,1],
    c = tmp_res.labels_
)
plt.show()


#%%
clustering.final_labels = tmp_res.labels_



#%%
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

#%%
def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

#%%
cah_fit = AgglomerativeClustering(n_clusters=10)

#%%
cah_fit = cah_fit.fit(clustering.kmeans_centers)

#%%
fig = plt.figure(1, figsize=(12, 7))
plot_dendrogram(cah_fit, labels = cah_fit.labels_)

#%%
cah_fit.labels_

#%%
tmp = Tools.read_np_picture('data/processed/models/autoencoder/train/k/20171109-192001.jpg',target_size = (27, 48), scale = 1/255)
tmp = tmp.reshape((1,27,48,3))
np.sum(model.get_encoded_prediction(tmp))

#%%
filenames = Tools.list_directory_filenames('data/processed/models/autoencoder/train/k/')
generator_imgs = Tools.generator_np_picture_from_filenames(filenames, target_size = (27, 48), batch = 10, nb_batch = 3, scale = 1/255)

predictions_list = []
predictions_id = []
for imgs in generator_imgs:
    predictions_id.append(imgs[0])
    predictions_list.append(model.get_encoded_prediction(imgs[1]))

#%%
np.concatenate(tuple(predictions_list), axis = 0)[0,:,:,:]

#%%
predictions_list[0][0,:,:,:]

#%%
print(pictures_preds[1,:,:,:])


#%%
pictures_preds.shape

#%%
