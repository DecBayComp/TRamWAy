
from .base import *
from inferencemap.spatial.scaler import *
from math import *
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans


class KMeansMesh(Voronoi):
	"""K-Means and Voronoi based tesselation."""
	def __init__(self, scaler=Scaler(), min_probability=None, avg_probability=None, **kwargs):
		Voronoi.__init__(self, scaler)
		#self.min_probability = min_probability
		#self.max_probability = None
		self.avg_probability = avg_probability
		#self.local_probability = None

	def _preprocess(self, points):
		points = Voronoi._preprocess(self, points)
		grid = RegularMesh(avg_probability=self.avg_probability)
		grid.tesselate(points)
		self._cell_centers = grid._cell_centers
		self.lower_bound = grid.lower_bound
		self.upper_bound = grid.upper_bound
		self.roi_subset_size = 10000
		self.roi_subset_count = 10
		return points

	def tesselate(self, points, tol=1e-3, plot=False, **kwargs):
		points = self._preprocess(points)
		self._cell_centers, _ = kmeans(np.asarray(points), self._cell_centers, \
			thresh=tol)
		if False:
			from sklearn.svm import OneClassSVM
			#self._postprocess()
			if self.roi_subset_size < points.shape[0]:
				permutation = np.random.permutation(points.shape[0])
				subsets = [ np.asarray(points)[permutation[i*self.roi_subset_size:(i+1)*self.roi_subset_size]] \
					for i in range(min(self.roi_subset_count, floor(points.shape[0]/self.roi_subset_size))) ]
			else:
				subsets = [np.asarray(points)]
			self.roi = []
			selected_centers = np.zeros(self._cell_centers.shape[0], dtype=bool)
			for subset in subsets:
				roi = OneClassSVM(nu=0.01, kernel='rbf', gamma=1, max_iter=1e5)
				roi.fit(subset)
				selected_centers = np.logical_or(selected_centers, roi.predict(self._cell_centers) == 1)
				self.roi.append(roi)
			if True:#plot:
				if points.shape[1] == 2:
					import matplotlib.pyplot as plt
					if isinstance(self.lower_bound, pd.DataFrame):
						x_ = 'x'
						y_ = 'y'
					else:
						x_ = 0
						y_ = 1
					xx, yy = np.meshgrid(np.linspace(self.lower_bound[x_], self.upper_bound[x_], 500), \
						np.linspace(self.lower_bound[y_], self.upper_bound[y_], 500))
					zz = self.roi[0].decision_function(np.c_[xx.ravel(), yy.ravel()])
					for roi in self.roi[1:]:
						zz = np.maximum(zz, roi.decision_function(np.c_[xx.ravel(), yy.ravel()]))
					zz = zz.reshape(xx.shape)
					subset = np.concatenate(subsets, axis=0)
					plt.plot(subset[:,0], subset[:,1], 'k.', markersize=8)
					plt.contour(xx, yy, zz, levels=[0], linewidths=2, colors='darkred')
					plt.scatter(self._cell_centers[selected_centers,0], self._cell_centers[selected_centers,1], c='blueviolet', s=40)
					plt.scatter(self._cell_centers[np.logical_not(selected_centers),0], self._cell_centers[np.logical_not(selected_centers),1], c='gold', s=40)
					plt.axis('equal')
					plt.show()
				else:
					raise AttributeError('can plot only 2D data')


