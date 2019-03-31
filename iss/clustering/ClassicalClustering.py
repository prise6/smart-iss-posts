from iss.clustering import AbstractClustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

class ClassicalClustering(AbstractClustering):

	def __init__(config, pictures_id, pictures_np):

		self.pca_fit = None
		self.pca_args = self.config['PCA']
		self.pca_reduction = None

		self.kmeans_fit = None
		self.kmeans_args = self.config['kmeans']
		self.kmeans_labels = None
		self.kmeans_centers = []

		self.cah_fit = None
		self.cah_args = self.config['CAH']
		self.cah_labels = None

		self.final_labels = None

		super().__init__(config, pictures_id, pictures_np)


	def pca_fit(self):

		self.pca_fit = PCA(**self.pca_args**)
		self.pca_fit.fit(self.pictures_np)
		self.pca_reduction = self.pca_fit.transform(self.pictures_np)

		return self

	def kmeans_fit(self):
		self.kmeans_fit = KMeans(self.kmeans_args**)
		self.kmeans_fit.fit(self.pca_reduction)
		self.kmeans_labels = self.kmeans_fit.labels_
		return self

	def compute_kmeans_centers(self):
		for cl in range(self.kmeans_args['n_clusters']):
			tmp = self.[np.where(self.kmeans_labels == cl)]
			self.kmeans_centers.append(np.mean(tmp, axis = 0))
		return self

	def cah_fit(self):

		self.cah_fit = AgglomerativeClustering(self.cah_args**)
		self.cah_fit.fit_predict(self.kmeans_centers)
		self.cah_labels = self.cah_fit.labels_
		return self

	def compute_cah_labels(self):
		self.final_labels = [self.cah_labels[old_cl] for old_cl in self.kmeans_labels]


