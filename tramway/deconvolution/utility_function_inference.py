# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: Jean-Baptiste Masson

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import numpy as np
import os
from   skimage import io
from   sklearn.model_selection import train_test_split

try: # recent Keras
	from   tensorflow.keras.preprocessing.image import ImageDataGenerator
	from   tensorflow.keras.callbacks import ModelCheckpoint
	from   tensorflow.keras.callbacks import ReduceLROnPlateau
except ImportError:
	from   keras.preprocessing.image import ImageDataGenerator
	from   keras.callbacks import ModelCheckpoint
	from   keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
def normalize_one_image(X):


	#nb_dim      = X[0,0,:].size
	min_X       = np.min(X, axis=(0,1))
	max_X       = np.max(X, axis=(0,1))

	d_min_max   = max_X - min_X
	X_minus_min = X - min_X[None,None] 
	X_out       = np.divide(X_minus_min,d_min_max[None,None])



	return X_out

#########################################################################################
#########################################################################################
#########################################################################################
def normalize_stack_whitening_from_trained_data(X, mean_value, mean_std):

	#nb_dim      = X[0,0,:].size

	#mean_X      = np.mean(X, axis=(1,2) , dtype=np.float32)
	#std_X       = np.std (X, axis=(1,2) , dtype=np.float32)

	#mean_mean_X = mean_X.mean()
	#mean_std_X  = std_X.mean() 

	X           = np.subtract(X,mean_value)
	X           = np.divide(X, mean_std)

	return X


#########################################################################################
#########################################################################################
#########################################################################################


#########################################################################################
#########################################################################################
#########################################################################################


#########################################################################################
#########################################################################################
#########################################################################################


#########################################################################################
#########################################################################################
#########################################################################################


#########################################################################################
#########################################################################################
#########################################################################################


#########################################################################################
#########################################################################################
#########################################################################################
