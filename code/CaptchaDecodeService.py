import os
import sys
import time

import numpy
import pickle
import theano
import theano.tensor as T
from theano.tensor.nnet import conv

from logistic_sgd_kaggle import LogisticRegression
from convolution_mlp_kaggle import LeNetConvPoolLayer
from mlp import HiddenLayer
from flask import Flask,jsonify,request,json
import logging
from logging import FileHandler

nkerns=[20, 50]
batch_size=50

new_char_map={3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',11:'A',\
  12:'B',13:'C',14:'D',15:'E',16:'F',17:'G',18:'H',19:'J',20:'K',21:'L',22:'M',23:'N',24:'P',25:'Q',26:'R',27:'S',28:'T',29:'U',30:'W',31:'X',32:'Y'}
rng = numpy.random.RandomState(23455)
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')   # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
					# [int] labels

ishape = (28, 28)  # this is the size of MNIST images

pkl_file=open('../ctcc_layers.pkl','rb')
W_0=pickle.load(pkl_file)
b_0=pickle.load(pkl_file)
W_1=pickle.load(pkl_file)
b_1=pickle.load(pkl_file)
W_2=pickle.load(pkl_file)
b_2=pickle.load(pkl_file)
W_3=pickle.load(pkl_file)
b_3=pickle.load(pkl_file)

# to a 4D tensor, compatible with our LeNetConvPoolLayer
layer0_input = x.reshape((batch_size, 1, 28, 28))
# Construct the first convolutional pooling layer:
# filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
# maxpooling reduces this further to (24/2,24/2) = (12,12)
# 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
		image_shape=(batch_size, 1, 28, 28),
		filter_shape=(nkerns[0], 1, 5, 5),W=W_0,b=b_0, poolsize=(2, 2))
# Construct the second convolutional pooling layer
# filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
# maxpooling reduces this further to (8/2,8/2) = (4,4)
# 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
		image_shape=(batch_size, nkerns[0], 12, 12),
		filter_shape=(nkerns[1], nkerns[0], 5, 5), W=W_1, b=b_1, poolsize=(2, 2))
# the TanhLayer being fully-connected, it operates on 2D matrices of
# shape (batch_size,num_pixels) (i.e matrix of rasterized images).
# This will generate a matrix of shape (20,32*4*4) = (20,512)
layer2_input = layer1.output.flatten(2)
# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * 4 * 4,
					 n_out=500,W=W_2,b=b_2, activation=T.tanh)
# classify the values of the fully-connected sigmoidal layer
layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=33,W=W_3,b=b_3)

def predict_load(predict_list):
	new_predict_set=list()
	new_predict_set.append(numpy.ndarray(shape=(batch_size,28*28), dtype=theano.config.floatX));
	index=0
	#pdb.set_trace()
	for code_array in predict_list:
		for pixel_index in xrange(0,28*28):
			new_predict_set[0][index][pixel_index] = code_array[pixel_index]/255;
  		index += 1;                            
		#pdb.set_trace()
	for i in xrange(index,50):
			new_predict_set[0][i]=new_predict_set[0][index-1]
	def shared_dataset(data_xy, borrow=True):
		data_x = data_xy[0]
		shared_x = theano.shared(numpy.asarray(data_x,
												dtype=theano.config.floatX),
									borrow=borrow)
		# shared_y = theano.shared(numpy.asarray(data_y,
												# dtype=theano.config.floatX),
									# borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
		return shared_x
	new_predict_set_x = shared_dataset(new_predict_set)	
	return new_predict_set_x

app = Flask(__name__)
@app.route('/', methods=['GET','POST'])
def process():
	if request.method == 'GET':
		return 'Index Page.'
	predict_codes=list()
	captcha_array=request.get_json()
	for captcha in captcha_array:
		predict_codes.append(captcha["captchaArray"])
	step = len(captcha_array)
	new_predict_set_x=predict_load(predict_codes)

	new_predict_model = theano.function([index], layer3.predict(),
		givens={
			x: new_predict_set_x[index * batch_size: (index + 1) * batch_size]})

	new_predict_res_array = new_predict_model(0)
	strs = '';
	for i in range(step):
		strs += new_char_map[new_predict_res_array[i]];
	return strs
if __name__ == '__main__':
	app.run(host='0.0.0.0',port=6800)
