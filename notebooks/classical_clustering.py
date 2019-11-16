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
generator_imgs = Tools.generator_np_picture_from_filenames(filenames, target_size = (27, 48), batch = 496, nb_batch = 10)

#%%
pictures_id, pictures_preds = Tools.encoded_pictures_from_generator(generator_imgs, model)

#%%
intermediate_output = pictures_preds.reshape((pictures_preds.shape[0], 3*6*16))


#%% [markdown]
# ## ACP
# Réduction de la dimension

#%%
clustering = ClassicalClustering(cfg.get('clustering')['classical'], pictures_id, intermediate_output)

#%%
clustering.compute_pca()


#%% [markdown]
# ## Kmeans
# Premiers clusters

#%%
clustering.compute_kmeans()
clustering.compute_kmeans_centers()

#%% [markdown]
# ## CAH
# Seconds clusters

#%%
clustering.compute_cah()
clustering.compute_cah_labels()

#%% [markdown]
# ## Résultats

#%% [markdown]
# ### Clusters intermediaires
#%%
fig = plt.figure(1, figsize=(12, 7))
plt.scatter(clustering.pca_reduction[:, 0], clustering.pca_reduction[:, 1], c = clustering.kmeans_labels)


#%% [markdown]
# ### Clusters finaux

#%%
plt.scatter(clustering.pca_reduction[:, 0], clustering.pca_reduction[:, 1], c = clustering.final_labels)


#%% [markdown]
# ### Sauvegarde des modèles

#%%
clustering.save()


#%%
# clustering = ClassicalClustering(cfg.get('clustering')['classical'])
clustering.load()

#%% [markdown]
# ## Visualisation des clusters

#%% 
def select_cluster(clustering, id_cluster):
    return [os.path.join('data/processed/models/autoencoder/train/k/', res[0] + '.jpg') for res in clustering.get_zip_results() if res[2] == id_cluster]

#%%
from IPython.display import Image

#%%
for cl in range(0,19):
    print("Cluster %s" % (cl))
    res_tmp = select_cluster(clustering, cl)
    print(len(res_tmp))
    image_array = [Tools.read_np_picture(f, target_size = (54, 96)) for f in res_tmp[:100]]
    # img = Tools.display_mosaic(image_array, nrow = 10)
    # fig = plt.figure(1, figsize=(12, 7))
    # plt.imshow(img, aspect = 'auto')
    # plt.show()


#%% [markdown]
# ## Zoom sur le cluster 0

#%%
res_tmp = select_cluster(clustering, 1)

#%%
print(len(res_tmp))
image_array = [Tools.read_np_picture(f, target_size = (54, 96)) for f in res_tmp]


#%%
Tools.display_mosaic(image_array, nrow = 18)


#%%
col = [1 if l == 1 else 0 for l in clustering.kmeans_labels]
plt.scatter(clustering.pca_reduction[:, 0], clustering.pca_reduction[:, 1], c = col)

#%%
plt.scatter(clustering.pca_reduction[np.array(col) == 1, 0], clustering.pca_reduction[np.array(col) == 1, 1])
