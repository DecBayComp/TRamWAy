# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: Jean-Baptiste Masson

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


# Import Libraries and model
## files management
import argparse
from   os.path import abspath
import gc
import os

## general
import numpy as np
import pandas as pd

## image
try:
    from skimage import io
except ImportError:
    raise ImportError('please install scikit-image>=0.14.2')
from os.path import split
from skimage.feature import peak_local_max
from skimage import data, img_as_float

## keras stuff
try: # recent Keras
	from tensorflow.keras.preprocessing.image import ImageDataGenerator
	from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
	#from tensorflow.keras.models import Model
	from .tf import Model
	from tensorflow.keras.layers import Input, Activation, UpSampling2D, Convolution2D, Convolution2DTranspose, MaxPooling2D, BatchNormalization, Dropout, Lambda, concatenate
	from tensorflow.keras import optimizers, losses
	from tensorflow.keras.utils import multi_gpu_model
	from tensorflow.keras.backend import clear_session
except ImportError:
	from keras.preprocessing.image import ImageDataGenerator
	from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
	from keras.models import Model
	from keras.layers import Input, Activation, UpSampling2D, Convolution2D, Convolution2DTranspose, MaxPooling2D, BatchNormalization, concatenate
	from keras import optimizers, losses
	from keras.utils import multi_gpu_model
	from keras.layers.core import Dropout, Lambda
	from keras.backend import clear_session
## local stuff
from .utility_function_inference import *

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
def get_parameters_mean_std(mean_std_file):

	file       = open(mean_std_file, "r")
	mean_image = np.float32(file.readline())
	std_image  = np.float32(file.readline())
	file.close()

	return mean_image,std_image
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
def preprocess_prediction(high_res_prediction,threshold):

	high_res_prediction[high_res_prediction<threshold] = 0
	image_high_res      = high_res_prediction.astype('uint16')

	image_high_res      = np.squeeze(image_high_res)
	high_res_prediction = np.squeeze(high_res_prediction)
	
	return high_res_prediction, image_high_res 


#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
def remove_anomalies(test):

	test_diff = np.diff(test,axis=0)
	test_diff = np.linalg.norm(test_diff, axis =1)
	II        = np.where(test_diff<=1.5)
	output    = np.delete(test, II, axis =0)

	return output

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
def dummy_get_rough_localisation_one_image(image_high_res, min_distance_peak,threshold_abs):

	(M,N) = image_high_res.shape
	liste   = []

	test = peak_local_max( image_high_res,threshold_abs=threshold_abs, min_distance=min_distance_peak)
	temp = test[:,0]
	test[:,[0, 1]] = test[:,[1, 0]]
	test = remove_anomalies(test)
	liste.append(test)
	del test, temp
	#del test
	return liste
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

def get_position_from_predicted_one_image(high_res_prediction, liste, marge, k):

	(M,N)        = high_res_prediction.shape
	xxx          = np.arange(M)
	xxx          = np.expand_dims(xxx,axis=1)
#	yyy          = xxx.transpose()

	yyy          = np.arange(N)
	yyy          = yyy.transpose()
	yyy          = np.expand_dims(yyy,axis=0)


	high_x       = np.multiply(xxx[np.newaxis,:],high_res_prediction[:,:])
	high_y       = np.multiply(yyy[np.newaxis,:],high_res_prediction[:,:])

	liste_output = []

#	for k in range(K):
		#print(k)
		#print("%i\t" % k)
	liste_loc      = liste[0]
	(KK, LL)       = liste_loc.shape
	point_high_res = np.zeros((KK,LL+1))
	if KK<=0:
		raise ValueError('no iteration; magnification might be too low')
	for kk in range(KK):
		ii        = liste_loc[kk,0]
		jj        = liste_loc[kk,1]

		min_x_loc = np.maximum(jj-marge,0)
		max_x_loc = np.minimum(jj+marge+1,M)
		min_y_loc = np.maximum( ii-marge, 0)
		max_y_loc = np.minimum(ii+marge+1, N)

		proba     = high_res_prediction[min_x_loc:max_x_loc, min_y_loc:max_y_loc]
		proba_x   = high_x[min_x_loc:max_x_loc, min_y_loc:max_y_loc]
		proba_y   = high_y[min_x_loc:max_x_loc, min_y_loc:max_y_loc]	

		#proba     = high_res_prediction[jj-marge:jj+marge+1, ii-marge:ii+marge+1]
		#proba_x   = high_x[jj-marge:jj+marge+1, ii-marge:ii+marge+1]
		sum_proba = np.sum(proba,axis=None)
			#print((k,kk,sum_proba,np.sum(proba_x,axis=None),np.sum(proba_y,axis=None) ))
			
		point_high_res[kk,0] = k
		point_high_res[kk,1] = np.sum(proba_x,axis=None)/sum_proba
		point_high_res[kk,2] = np.sum(proba_y,axis=None)/sum_proba

			#print((liste_loc[kk,0],point_high_res[kk,1],liste_loc[kk,1],point_high_res[kk,2]))
	point_high_res[:,[1,2]] = point_high_res[:,[2, 1]]
	liste_output.append(point_high_res)
	del proba, proba_x, proba_y, sum_proba, liste_loc, point_high_res
	#print(K)
	#if K==1:
	#	point_high_res[:,0] = nb_out

	position         = pd.DataFrame(np.concatenate(liste_output))
	position.dropna(how='all')
	position.columns = ['nb','x','y']
	position['nb']   = position['nb'].astype(int)
	
	return position
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

def print_position_files(position,path,root_name, bool_header):

	#file = open(path + '/' + 'position_' root_name + '.txt','w+' )
	full_path= path + '/' + 'position_' + root_name + '.txt'
	if bool_header:
		position.to_csv(full_path,index=None, sep=',') 
	else:
		position.to_csv(full_path,index=None,header=None, sep=',') 

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
def save_trimmed_original_image_magnified_for_testing_purposes(stack,magnification,n):

	path_split           = os.path.split(stack)
	name_split           = os.path.splitext(path_split[1])

	Images = io.imread(stack)
	Images = Images[:,n:-n,n:-n]
	# get dataset dimensions
	(K, M, N) = Images.shape
	high_res = np.zeros((K,M*magnification, N*magnification))
	for i in range(K):
		high_res[i,:,:] = np.kron(Images[i,:,:], np.ones((magnification,magnification)))  
	(K, M, N) = high_res.shape
	image_high_res = high_res.astype('uint16')

	io.imsave(path_split[0] + '/' + name_split[0] + '_magnified.tiff', image_high_res)


	return 1

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################


def pre_process_one_image(Image, index_split, M_extend, N_extend, M_original, N_original, magnification,mean_image,std_image):

	#(M,N)    = Images.shape
	Image_extend      = np.zeros((M_extend,N_extend))
	Image_extend[0:M_original,0:N_original] = Image
	liste_Image_extend_magnificaton = []
	for x in index_split:
		high_res = np.zeros((M_extend*magnification, N_extend*magnification))
		high_res = np.kron(  Image_extend[x[0]:x[1],x[2]:x[3]], np.ones((magnification,magnification)))
		high_res = normalize_one_image(high_res)
		high_res = normalize_stack_whitening_from_trained_data(high_res, mean_image, std_image)
		liste_Image_extend_magnificaton.append(high_res)

	Image_to_stack = np.asarray(liste_Image_extend_magnificaton)
	(K_stack, _,_) = Image_to_stack.shape	


	return Image_to_stack, K_stack

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

def reassemble_one_image(Image_to_stack,K_stack, empty_Image_high_res, index_reconstitute_mag, M_cut, N_cut, n_high):

	liste_loc = np.split(Image_to_stack,K_stack)

	for indice,x in enumerate(index_reconstitute_mag,0):
		#print(x[0],x[1],x[2],x[3])
		empty_Image_high_res[x[0]:x[1],x[2]:x[3]] = np.squeeze(liste_loc[indice][0,n_high:-n_high,n_high:-n_high])
	empty_Image_high_res                      = empty_Image_high_res[0:M_cut,0:N_cut]


	return empty_Image_high_res

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################


def define_all_indexes_for_image_slicing(Images_example, M_theo, N_theo, n, magnification):

	##################################################################
	(M,N)             = Images_example.shape
	II_max            = int(np.ceil((M-M_theo)/(M_theo-2*n)))+1
	JJ_max            = int(np.ceil((N-N_theo)/(N_theo-2*n)))+1
	M_extend          = (II_max ) * M_theo - n
	N_extend          = (JJ_max ) * N_theo - n
	M_extend_high_res = M_extend*magnification 
	N_extend_high_res = N_extend*magnification 
	##################################################################
	I_start           = np.arange(0,II_max)*(M_theo-2*n)
	I_end             = I_start + M_theo
	J_start           = np.arange(0,JJ_max)*(N_theo-2*n) 
	J_end             = J_start + N_theo
	################################################################
	I_start_rec       = np.arange(0,II_max)*(M_theo-2*n)
	I_end_rec         = I_start_rec + (M_theo-2*n)
	J_start_rec       = np.arange(0,JJ_max)*(N_theo-2*n)
	J_end_rec         = J_start_rec + (N_theo-2*n)
	###############################################################
	I_start_rec_mag   = np.arange(0,II_max)*(M_theo-2*n)*magnification
	I_end_rec_mag     = I_start_rec_mag + (M_theo-2*n)*magnification
	J_start_rec_mag   = np.arange(0,JJ_max)*(N_theo-2*n)*magnification
	J_end_rec_mag     = J_start_rec_mag + (N_theo-2*n)*magnification
	##################################################################
	index_split       = []
	index_reconstitute = []
	index_reconstitute_mag = []
	
	for i in range(II_max):
		for j in range(JJ_max):
			index_split.append([I_start[i], I_end[i], J_start[j],J_end[j]])
			index_reconstitute.append([I_start_rec[i], I_end_rec[i], J_start_rec[j],J_end_rec[j]])
			index_reconstitute_mag.append([I_start_rec_mag[i], I_end_rec_mag[i], J_start_rec_mag[j],J_end_rec_mag[j]])
	##################################################################
	M_cut_high_res = M*magnification - 2*n*magnification
	N_cut_high_res = N*magnification - 2*n*magnification
	n_high         = n*magnification
	##################################################################
	


	return M_extend, N_extend,M_extend_high_res, N_extend_high_res, index_split, index_reconstitute, index_reconstitute_mag, M_cut_high_res, N_cut_high_res,n_high 
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
def Inference(stack,magnification,  weights, mean_image, std_image, M_theo, N_theo,n, threshold, min_distance_peak, threshold_abs, marge, bool_GPU ):
	## name of the stack with full path
	## factor of magnification
	## full path to wieghts file
	## mean of the training images
	## std of the training images

	Images = io.imread(stack)
	(K_original,M_original,N_original) = Images.shape
	if bool_GPU:
		model = generate_network_keras_multi_GPUs((M_theo*magnification, N_theo*magnification, 1))
	else:
		model = generate_network_keras_for_inference_one_GPU((M_theo*magnification, N_theo*magnification, 1))

	model.load_weights(weights)

	## all usefull indexes
	M_extend, N_extend,M_extend_high_res, N_extend_high_res, index_split,\
	index_reconstitute, index_reconstitute_mag, M_cut_high_res, N_cut_high_res, \
	n_high = define_all_indexes_for_image_slicing(np.squeeze(Images[0,:,:]), M_theo, N_theo, n, magnification)
	## empty recipient
	empty_Image_high_res = np.zeros((M_extend_high_res - 2*n*magnification , N_extend_high_res - 2*n*magnification))

	##
	liste_position = []
	#high_res_prediction = np.zeros((K_original, M_cut_high_res, N_cut_high_res))
	#liste_high_res = []
	for i in range(K_original):
		if i%100==0:
			print(i)
		Image          		                = np.squeeze(Images[i,:,:])
		Image_to_stack,K_stack              = pre_process_one_image(Image, index_split, M_extend, N_extend, M_original, N_original, magnification,mean_image,std_image)
		Image_to_stack                      = np.squeeze(model.predict(np.expand_dims(Image_to_stack,axis=3), batch_size=1))
		high_res                            = reassemble_one_image(Image_to_stack,K_stack, empty_Image_high_res, index_reconstitute_mag, M_cut_high_res, N_cut_high_res, n_high)
		high_res, image_high_res            = preprocess_prediction(high_res,threshold)
		liste_low_res                       = dummy_get_rough_localisation_one_image(image_high_res, min_distance_peak,threshold_abs)
		position                            = get_position_from_predicted_one_image(high_res, liste_low_res, marge,i)
		#position                            = get_position_from_predicted(np.expand_dims(high_res,axis=0), liste_low_res, marge,i)
		liste_position.append(position)
		#high_res_prediction[i,:,:] = high_res
		#liste_high_res.append(high_res)
		del Image, Image_to_stack, K_stack, high_res, image_high_res,liste_low_res, position

	#high_res_presdiction = np.stack(liste_high_res)
	position = pd.concat(liste_position)
	#del high_res_norm
	high_res_presdiction = None

	return high_res_prediction, position
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
def generate_network_keras_for_inference_one_GPU(input_dim):

    input_         = Input (shape = (input_dim))
    full_CNN       = CNN(input_,'CNN')
    model          = Model (inputs= input_, outputs=full_CNN)
    opt            = optimizers.Adam(lr= 0.001)
    model.compile(optimizer=opt, loss = Simplest_loss(input_dim))
  

    return model


#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
def generate_network_keras_multi_GPUs(input_dim):
    
    input_         = Input (shape = (input_dim))
    full_CNN       = CNN(input_,'CNN')
    model          = Model (inputs= input_, outputs=full_CNN)
    parallel_model = multi_gpu_model(model,gpus=4)
    opt            = optimizers.Adam(lr= 0.001)
 
    parallel_model.compile(optimizer=opt, loss = Simplest_loss(input_dim))
    #model.compile(optimizer=opt, loss = Simplest_loss(input_dim))

    return parallel_model

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################


def CNN(input,names):

    # start 
    Features1  = module_CNN(16,5,5,names+'F1',0)(input)
    Features2  = module_CNN(16,5,5,names+'F2',0)(Features1)
    # reduce size convole
    Features3  = module_CNN_to_down(32,5,5,names+'F3',0)(Features2)
    Features4  = module_CNN_to_down(64,5,5,names+'F4',0)(Features3)
    Features5  = module_CNN_to_down(128,5,5,names+'F5',0)(Features4)
    Features6  = module_CNN_to_down(256,5,5,names+'F6',0)(Features5)
    # convolve
    Features7  = module_CNN(512,5,5,names+'F7',1)(Features6)
    # increase size concatenae covolve
    Features8  = module_CNN_to_up(256,5,5,names+'F8',1)(Features7)
    Features9  = concatenate([Features8, Features5])
   
    Features10  = module_CNN_to_up(128,5,5,names+'F9',1)(Features9)
    Features11  = concatenate([Features10, Features4])

    Features12  = module_CNN_to_up(64,5,5,names+'F10',1)(Features11)
    Features13 = concatenate([Features12, Features3])

    Features14  = module_CNN_to_up(32,5,5,names+'F11',1)(Features13)
    Features15 = concatenate([Features14, Features2])
    #convolve    
    Features16 = module_CNN(16,5,5,names+'F12',0)(Features15)
    Features17 = module_CNN(16,5,5,names+'F13',0)(Features16)
    #get_densities
    full_CNN   = Convolution2D(1, kernel_size=(1, 1), strides=(1, 1), padding="same",\
                                  activation="linear", use_bias = False,\
                                  kernel_initializer="Orthogonal",name='Prediction')(Features17)

    return full_CNN

#########################################################################################
#########################################################################################
#########################################################################################
def module_CNN_to_down(nb_filter, rk, ck, name, boolean_dropout):
    def f(input):
        layer                = Convolution2D(nb_filter, kernel_size=(rk, ck), strides=(1,1),\
                               padding="same", use_bias=False,\
                               kernel_initializer="Orthogonal",name='conv-'+name)(input)
        layer_norm           = BatchNormalization(name='BN-'+name)(layer)
        layer_norm_relu      = Activation(activation = "relu",name='Relu-'+name)(layer_norm)
        if boolean_dropout:
            layer_norm_relu =  Dropout(0.1) (layer_norm_relu)
        layer_norm_relu_pool = MaxPooling2D(pool_size=(2,2),name=name+'Pool1')(layer_norm_relu)


        return layer_norm_relu_pool
    return f

#########################################################################################
#########################################################################################
#########################################################################################

def module_CNN(nb_filter, rk, ck, name, boolean_dropout):
    def f(input):
        layer           = Convolution2D(nb_filter, kernel_size=(rk, ck), strides=(1,1),\
                               padding="same", use_bias=False,\
                               kernel_initializer="Orthogonal",name='conv-'+name)(input)
        layer_norm      = BatchNormalization(name='BN-'+name)(layer)
        layer_norm_relu = Activation(activation = "relu",name='Relu-'+name)(layer_norm)
        if boolean_dropout:
            layer_norm_relu =  Dropout(0.1) (layer_norm_relu)

        return layer_norm_relu
    return f
#########################################################################################
#########################################################################################
#########################################################################################
def module_CNN_to_up(nb_filter, rk, ck, name, boolean_dropout):
    def f(input):
        layer           = Convolution2DTranspose(nb_filter, kernel_size=(rk, ck), strides=(2,2),\
                               padding="same", use_bias=False,\
                               kernel_initializer="Orthogonal",name='conv-'+name)(input)
        layer_norm      = BatchNormalization(name='BN-'+name)(layer)
        layer_norm_relu = Activation(activation = "relu",name='Relu-'+name)(layer_norm)
        if boolean_dropout:
            layer_norm_relu =  Dropout(0.1) (layer_norm_relu)

        return layer_norm_relu
    return f
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

def Simplest_loss(input_shape):
    def MSE_densities(ground_truth, prediction):

        loss_loc = 10*losses.mean_squared_error(ground_truth,prediction)
        
        return loss_loc
    return MSE_densities

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################


if __name__ == '__main__':

	## parameters hard coded
	magnification     = 10
	threshold         = 0
	min_distance_peak = 2
	marge             = 3
	bool_header       = 1
	threshold_abs     = 1.
	M_theo            = 64
	N_theo            = M_theo
	n                 = 2 

	## clean evrything
	##
	gc.collect()
	clear_session()
	##
	## get the prameters from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('--stack',    help="path to tiff stack")
	parser.add_argument('--weights',  help="name of the weights file")
	parser.add_argument('--nb_GPU',   type=int, help="boolean 0= 1 GPU , 1 = multiple GPU")
	parser.add_argument('--mean_std', help="path to the mean_std.txt file")
	args   = parser.parse_args()
	##
	##

	path_split              = os.path.split(abspath(args.stack))
	name_split              = os.path.splitext(path_split[1])
	path_weights            = abspath(args.weights)
	mean_std_file           = abspath(args.mean_std)
	bool_GPU                = bool(args.nb_GPU)
	stack                   = abspath(args.stack)

    ## at the moment it is in separate files it could be stored depending on the use 
	mean_image,std_image    = get_parameters_mean_std(mean_std_file)

	## to be commented when applied in tramway 
	save_trimmed_original_image_magnified_for_testing_purposes(stack,magnification,n)

	## when in tramway it is not necesary to output high_res_prediction , is might be usefull anyway so to be put in option
	high_res_prediction, position = Inference(stack,magnification,  path_weights, mean_image, std_image, M_theo, N_theo,n, threshold, min_distance_peak, threshold_abs, marge,bool_GPU )
	
	## to be extended to rw files
	print_position_files(position,path_split[0],name_split[0], bool_header)

	## just for talks output high resolution version of the raw data with the distribution of position
	#image                   = high_res_prediction.astype('uint16')
	#io.imsave(path_split[0] + '/' + 'predicted.tiff', image)



#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
