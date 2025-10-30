'''
******************************************************************************************
  Assembly:                Mathy
  Filename:                Clusters.py
  Author:                  Terry D. Eppler
  Created:                 05-31-2022

  Last Modified By:        Terry D. Eppler
  Last Modified On:        05-01-2025
******************************************************************************************
<copyright file="Clusters.py" company="Terry D. Eppler">

     Mathy Clusters

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the “Software”),
 to deal in the Software without restriction,
 including without limitation the rights to use,
 copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software,
 and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.

 You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

</copyright>
<summary>
	Clusters.py
</summary>
******************************************************************************************
'''
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from Booger import Error, ErrorDialog
from sklearn.cluster import (KMeans, DBSCAN, MeanShift, AffinityPropagation,
                             SpectralClustering, AgglomerativeClustering,
                             Birch, OPTICS)
from sklearn.metrics import silhouette_score


class Cluster( ):
	"""

        Purpose:
        ---------
		Abstract base class for clustering models, including methods
		for fitting, predicting, evaluating, and visualization.

	"""

	def fit( self, X: np.ndarray ) -> None:
		"""

	        Purpose:
	        ---------
			Fit the clustering model to the stores.

			Parameters:
			----------
			X: The input stores of shape (n_samples, n_features).

		"""
		raise NotImplementedError( )

	def predict( self, X: np.ndarray ) -> np.ndarray:
		"""

	        Purpose:
	        ---------
			Predict the cluster labels for the input stores.

			Parameters:
			----------
			X: Input features to cluster.

			Returns:
			---------
			np.ndarray

		"""
		raise NotImplementedError( )

	def evaluate( self, X: np.ndarray ) -> float:
		"""

	        Purpose:
	        ---------
			Evaluate clustering performance using silhouette score.

			Parameters:
			----------
			X: Input features to cluster.

			Returns:
			---------
			float

		"""
		raise NotImplementedError( )

	def visualize_clusters( self, X: np.ndarray ) -> None:
		"""

	        Purpose:
	        ---------
			Visualize clusters using a 2D scatter plot.

			Parameters:
			----------
			X: Input stores of shape (n_samples, 2).

		"""
		raise NotImplementedError( )


class KMeansClustering( Cluster ):
	"""

		Purpose:
		---------
		The KMeans algorithm clusters stores by trying to separate samples in n groups of equal
		variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares
		(see below). This algorithm requires the number of clusters
		to be specified. It scales well to large number of samples and has been used across a
		large range of application areas in many different fields.

		The algorithm has three steps. The first step chooses the initial centroids,
		with the most basic method being to choose samples from the dataset. After initialization,
		K-means consists of looping between the two other steps. The first step assigns each sample
		to its nearest centroid. The second step creates new centroids by taking the mean value of
		all of the samples assigned to each previous centroid. The difference between the old and
		the new centroids are computed and the algorithm repeats these last two steps until this
		value is less than a threshold. In other words, it repeats until the centroids do not move
		significantly.

	"""

	def __init__( self, num: int = 8, rando: int = 42 ) -> None:
		"""
			Purpose:
			---------
			Initialize the KMeans model.

			Parameters:
			----------
			num: Number of clusters to form.
			rando: Random seed for reproducibility.
			rando: int

		"""
		super( ).__init__( )
		self.model = KMeans( n_clusters=num, random_state=rando )


	def fit( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Fit the KMeans model on the dataset.

			Parameters:
			----------
			X: The input stores.

		"""
		try:
			self.model.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'KMeansClustering'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def predict( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Predict the closest cluster each sample in X belongs to.

			Parameters:
			----------
			X: The input stores.

			Returns:
			--------
			np.ndarray

		"""
		try:
			return self.model.predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'KMeansClustering'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def evaluate( self, X: np.ndarray ) -> Optional[ float ]:
		"""

			Purpose:
			---------
			Evaluate clustering performance using silhouette score.

			Parameters:
			----------
			X: The input stores.
			X: np.ndarray

			Returns:
			---------
			float

		"""
		try:
			labels = self.model.predict( X )
			return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'KMeansClustering'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def visualize_clusters( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Visualize clustering result using a scatter plot.

			Parameters:
			----------
			X: Input stores of shape (n_samples, 2).

		"""
		try:
			if X is None:
				raise Exception( 'The input arguement "X" is required.' )
			else:
				labels = self.model.predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'viridis' )
				plt.title( "KMeans Cluster" )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'KMeansClustering'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class DbscanClustering( Cluster ):
	"""

		Purpose:
		---------
		The DBSCAN algorithm views clusters as areas of high density separated by areas of low
		density. Due to this rather generic view, clusters found by DBSCAN can be any shape,
		as opposed to k-means which assumes that clusters are convex shaped. The central component
		to the DBSCAN is the concept of core samples, which are samples that are in areas of high
		density. A cluster is therefore a set of core samples, each close to each other (measured
		by some distance measure) and a set of non-core samples that are close to a core sample
		(but are not themselves core samples). There are two parameters to the algorithm,
		min_samples and eps, which define formally what we mean when we say dense. Higher
		min_samples or lower eps indicate higher density necessary to form a cluster.

	"""

	def __init__( self, eps: float = 0.5, min: int = 5 ) -> None:
		"""

			Purpose:
			---------
			Initialize the DBSCAN model.

			Parameters:
			----------
			eps: float
			min: int

		"""
		super( ).__init__( )
		self.model = DBSCAN( eps=eps, min_samples=min )


	def fit( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Fit the DBSCAN model to the stores.

			Parameters:
			----------
			X: Input features.

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.model.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DbscanClustering'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def predict( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Predict clusters using DBSCAN fit.

			Parameters:
			----------
			X: Input features.

			Returns:
			---------
			np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				return self.model.fit_predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DbscanClustering'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def evaluate( self, X: np.ndarray ) -> Optional[ float ]:
		"""

			Purpose:
			---------
			Evaluate DBSCAN clusters with silhouette score.

			Parameters:
			----------
			X: Input features.
			X: np.ndarray

			Returns:
			---------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				return silhouette_score( X, labels ) if len( set( labels ) ) > 1 else -1.0
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DbscanClustering'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def visualize_clusters( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Visualize DBSCAN clusters.

			Parameters:
			----------
			X: 2D input features.

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'plasma' )
				plt.title( 'DBSCAN Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'DbscanClustering'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class AgglomerativeClusteringModel( Cluster ):
	"""

		Purpose:
		---------
		The AgglomerativeClustering object performs a hierarchical clustering using a
		bottom up approach: each observation starts in its own cluster, and clusters are
		successively merged together. The linkage criteria determines the metric used for the merge
		strategy:

			Ward minimizes the sum of squared differences within all clusters. It is a
			variance-minimizing approach and in this sense is similar to the k-means objective
			function but tackled with an agglomerative hierarchical approach.

			Maximum or complete linkage minimizes the maximum distance between observations of
			pairs of clusters.

			Average linkage minimizes the average of the distances between all observations of
			pairs of clusters.

		Single linkage minimizes the distance between the closest observations of pairs of clusters.
		AgglomerativeClustering can also scale to large number of samples when it is used jointly
		with a connectivity matrix, but is computationally expensive when no connectivity
		constraints are added between samples: it considers at each step all the possible merges.

	"""

	def __init__( self, num: int = 2 ) -> None:
		"""

			Purpose:
			---------
			Initialize AgglomerativeClustering.

			Parameters:
			----------
			num: int

		"""
		super( ).__init__( )
		self.model = AgglomerativeClustering( n_clusters=num )

	def fit( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Fit Agglomerative model to stores.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.model.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AgglomerativeClusteringModel'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

	def predict( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Predict clusters using agglomerative clustering.

			Parameters:
			----------
			X: np.ndarray

			Returns:
			---------
			np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				return self.model.fit_predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AgglomerativeClusteringModel'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def evaluate( self, X: np.ndarray ) -> Optional[ float ]:
		"""

			Purpose:
			---------
			Evaluate agglomerative clustering using silhouette score.

			Parameters:
			----------
			X: np.ndarray

			Returns:
			-------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AgglomerativeClusteringModel'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def visualize_clusters( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Visualize agglomerative clustering results.

			Parameters:
			----------
			X: 2D input stores.
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'tab10' )
				plt.title( 'Agglomerative Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AgglomerativeClusteringModel'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

class SpectralClusteringModel( Cluster ):
	"""

		Purpose:
		---------
		SpectralClustering does a low-dimension embedding of the affinity matrix between samples,
		followed by a KMeans in the low dimensional space. It is especially efficient if the
		affinity matrix is sparse and the pyamg module is installed. SpectralClustering requires
		the number of clusters to be specified. It works well for a small number of clusters but
		is not advised when using many clusters.

		For two clusters, it solves a convex relaxation of the normalised cuts problem on the
		similarity graph: cutting the graph in two so that the weight of the edges cut is small
		compared to the weights of the edges inside each cluster. This criteria is especially
		interesting when working on images: graph vertices are pixels, and edges of the similarity
		graph are a function of the gradient of the image.

	"""

	def __init__( self, num: int = 8 ) -> None:
		"""

			Purpose:
			---------
			Initialize the SpectralClustering model.

			Parameters:
			----------
			num: int

		"""
		super( ).__init__( )
		self.model = SpectralClustering( n_clusters=num )

	def fit( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Fit the SpectralClustering model.

			Parameters:
			----------
			X:  np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.model.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SpectralClusteringModel'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )

	def predict( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Predict clusters using SpectralClustering.

			Parameters:
			----------
			X: np.ndarray


			Return:
			--------
			np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				return self.model.fit_predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SpectralClusteringModel'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

	def evaluate( self, X: np.ndarray ) -> Optional[ float ]:
		"""

			Purpose:
			---------
			Evaluate SpectralClustering results.

			Parameters:
			----------
			X: np.ndarray

			Return:
			--------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SpectralClusteringModel'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def visualize_clusters( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Visualize Spectral Cluster results.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'Accent' )
				plt.title( 'Spectral Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'SpectralClusteringModel'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class MeanShiftClustering( Cluster ):
	"""

		Purpose:
		---------
		MeanShift clustering aims to discover blobs in a smooth density of samples.
		It is a centroid based algorithm, which works by updating candidates for centroids to be
		the mean of the points within a given region. These candidates are then filtered in a
		post-processing stage to eliminate near-duplicates to form the final set of centroids.

		The algorithm automatically sets the number of clusters, instead of relying on a parameter
		bandwidth, which dictates the size of the region to search through. This parameter can be
		set manually, but can be estimated using the provided estimate_bandwidth function, which
		is called if the bandwidth is not set.

		The algorithm is not highly scalable, as it requires multiple nearest neighbor searches
		during the execution of the algorithm. The algorithm is guaranteed to converge,
		however the algorithm will stop iterating when the change in centroids is small.

	"""

	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize MeanShift model.

		"""
		super( ).__init__( )
		self.model = MeanShift( )


	def fit( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Fit MeanShift model to the stores.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.model.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MeanShiftClustering'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def predict( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Predict clusters using MeanShift.

			Parameters:
			----------
			X: np.ndarray

			Return:
			--------
			np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				return self.model.fit_predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MeanShiftClustering'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )

	def evaluate( self, X: np.ndarray ) -> Optional[ float ]:
		"""

			Purpose:
			---------
			Evaluate MeanShift clustering.

			Parameters:
			----------
			X:  np.ndarray

			Return:
			--------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MeanShiftClustering'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def visualize_clusters( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Visualize MeanShift clustering.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'Set1' )
				plt.title( 'MeanShift Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'MeanShiftClustering'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class AffinityPropagationClustering( Cluster ):
	"""

		Purpose:
		---------
		AffinityPropagation creates clusters by sending messages between pairs of samples until
		convergence. A dataset is then described using a small number of exemplars, which are
		identified as those most representative of other samples. The messages sent between pairs
		represent the suitability for one sample to be the exemplar of the other, which is updated
		in response to the values from other pairs. This updating happens iteratively until
		convergence, at which point the final exemplars are chosen,
		and hence the final clustering is given.

	"""

	def __init__( self ) -> None:
		"""

			Purpose:
			---------
			Initialize AffinityPropagation model.

		"""
		super( ).__init__( )
		self.model = AffinityPropagation( )


	def fit( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Fit the model to stores.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.model.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AffinityPropagationClustering'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def predict( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Predict clusters.

			Parameters:
			----------
			X: np.ndarray

			Returns:
			--------
			np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				return self.model.fit( X ).labels_
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AffinityPropagationClustering'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def evaluate( self, X: np.ndarray ) -> Optional[ float ]:
		"""

			Purpose:
			---------
			Evaluate clustering result.

			Parameters:
			----------
			X: np.ndarray

			Return:
			--------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit( X ).labels_
				return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AffinityPropagationClustering'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def visualize_clusters( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Visualize clustering with AffinityPropagation.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit( X ).labels_
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'Paired' )
				plt.title( 'AffinityPropagation Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'AffinityPropagationClustering'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class BirchClustering( Cluster ):
	"""

		Purpose:
		---------
		The Birch builds a tree called the Clustering Feature Tree (CFT) for the given stores.
		The stores is essentially lossy compressed to a set of Clustering Feature nodes (CF Nodes).
		The CF Nodes have a number of subclusters called Clustering Feature subclusters
		(CF Subclusters) and these CF Subclusters located in the non-terminal
		CF Nodes can have CF Nodes as children.

		The BIRCH algorithm has two parameters, the threshold and the branching factor.
		The branching factor limits the number of subclusters in a node and the threshold limits
		the distance between the entering sample and the existing subclusters.

		This algorithm can be viewed as an instance or stores reduction method, since it reduces the
		input stores to a set of subclusters which are obtained directly from the leaves of the CFT.
		This reduced stores can be further processed by feeding it into a global clusterer.
		This global clusterer can be set by n_clusters. If n_clusters is set to None,
		the subclusters from the leaves are directly read off, otherwise a global clustering step
		labels these subclusters into global clusters (labels) and the samples are
		mapped to the global label of the nearest subcluster.

	"""

	def __init__( self, n_clusters: Optional[ int ] = None ) -> None:
		"""

			Purpose:
			---------
			Initialize Birch clustering.

			Parameters:
			----------
			num: Optional[int]

		"""
		super( ).__init__( )
		self.model = Birch( n_clusters=n_clusters )


	def fit( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Fit Birch clustering model.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.model.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BirchClustering'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def predict( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Predict clusters with Birch.

			Parameters:
			----------
			X: np.ndarray

			Returns:
			--------
			np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				return self.model.predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BirchClustering'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def evaluate( self, X: np.ndarray ) -> Optional[ float ]:
		"""

			Purpose:
			---------
			Evaluate Birch clustering.

			Parameters:
			----------
			X: np.ndarray

			Return:
			--------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.predict( X )
				return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BirchClustering'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def visualize_clusters( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Visualize Birch clustering.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'Dark2' )
				plt.title( 'Birch Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'BirchClustering'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


class OpticsClustering( Cluster ):
	"""

		Purpose:
		---------
		The OPTICS is a generalization of DBSCAN that relaxes the eps requirement from a single
		value to a value range. The key difference between DBSCAN and OPTICS is that the OPTICS
		algorithm builds a reachability graph, which assigns each sample both a reachability_
		distance, and a spot within the cluster ordering_ attribute; these two attributes are
		assigned when the model is fitted, and are used to determine cluster membership.

		If OPTICS is run with the default value of inf set for max_eps, then DBSCAN style
		cluster extraction can be performed repeatedly in linear time for any given eps value
		using the cluster_optics_dbscan method. Setting max_eps to a lower value will result
		in shorter run times, and can be thought of as the maximum neighborhood radius from
		each point to find other potential reachable points.

	"""

	def __init__( self, min: int = 5 ) -> None:
		"""

			Purpose:
			---------
			Initialize OPTICS model.

			Parameters:
			----------
			min: int

		"""
		super( ).__init__( )
		self.model = OPTICS( min_samples=min )

	def fit( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Fit OPTICS model.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				self.model.fit( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OpticsClustering'
			exception.method = 'fit( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )


	def predict( self, X: np.ndarray ) -> np.ndarray | None:
		"""

			Purpose:
			---------
			Predict clusters with OPTICS.

			Parameters:
			----------
			X: np.ndarray

			Return:
			--------
			np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				return self.model.fit_predict( X )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OpticsClustering'
			exception.method = 'predict( self, X: np.ndarray ) -> np.ndarray'
			error = ErrorDialog( exception )
			error.show( )


	def evaluate( self, X: np.ndarray ) -> Optional[ float ]:
		"""

			Purpose:
			---------
			Evaluate OPTICS clustering.

			Parameters:
			----------
			X: Input features.
			X: np.ndarray

			Returns:
			---------
			float

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				return silhouette_score( X, labels )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OpticsClustering'
			exception.method = 'evaluate( self, X: np.ndarray ) -> float'
			error = ErrorDialog( exception )
			error.show( )


	def visualize_clusters( self, X: np.ndarray ) -> None:
		"""

			Purpose:
			---------
			Visualize OPTICS clustering result.

			Parameters:
			----------
			X: np.ndarray

		"""
		try:
			if X is None:
				raise Exception( 'The input argument "X" is required.' )
			else:
				labels = self.model.fit_predict( X )
				plt.scatter( X[ :, 0 ], X[ :, 1 ], c = labels, cmap = 'rainbow' )
				plt.title( 'OPTICS Cluster' )
				plt.show( )
		except Exception as e:
			exception = Error( e )
			exception.module = 'Mathy'
			exception.cause = 'OpticsClustering'
			exception.method = 'visualize_clusters( self, X: np.ndarray ) -> None'
			error = ErrorDialog( exception )
			error.show( )