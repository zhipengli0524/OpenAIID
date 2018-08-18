#!/usr/bin/env python
# -*- coding:utf8 -*-

from layers import *
from deepid_class import *

import cPickle
import gzip
import os
import sys
import time
import numpy
import theano
import logging

class DeepIDGenerator:
    def __init__(self, n_hidden=160,
                       nkerns=[20,40,60,80],
                       params_file=None):

        self._rng = numpy.random.RandomState(1234)
        self._nkerns = nkerns
        self._n_hidden = n_hidden

        if params_file is not None:
            self._params_file = params_file
        else:
            file_dir = os.path.dirname(__file__)
            file_dir = os.path.abspath(file_dir)
            self._params_file = os.path.join(file_dir, "model_params", "params.bin")
            logging.info("No params file given, using default: %s" % self._params_file)

        self._exist_params = None
        self._initialized = False

    def isInitialized(self):
        """return initialized state
        Args:
            self
        Returns:
            bool    True or False
        Raises:
            None
        """
        return self._initialized

    def initialize(self):
        """initialize the genarator
        Args:
        Returns:
        Raises:
        """
        pd_helper = ParamDumpHelper(self._params_file)
        exist_params = pd_helper.get_params_from_file()
        if len(exist_params) != 0:
            self._exist_params = exist_params[-1]
        else:
            raise RuntimeError('no trained params found in %s' % self._params_file)

        self._layer_params(self._nkerns, 1)
        self._build_layer_architecture(self._n_hidden, acti_func=relu)
        self._initialized = True

    def generate_deepid(self, input_vec):
        """generate deepid
        Args:
            input_vec    numpy.ndarray
        Returns:
            output_vec   numpy.ndarray
        Raises:
            TyepError
        """
        if not isinstance(input_vec, numpy.ndarray):
            raise TypeError("input_vec should be type of numpy.ndarray")

        logging.info('id generating ...')
        deepid_data = self.generator([input_vec])
        return deepid_data[0]

    def _layer_params(self, nkerns, batch_size):
        src_channel = 3
        self.layer1_image_shape  = (batch_size, src_channel, 55, 47)
        self.layer1_filter_shape = (nkerns[0],  src_channel, 4, 4)
        self.layer2_image_shape  = (batch_size, nkerns[0], 26, 22)
        self.layer2_filter_shape = (nkerns[1], nkerns[0], 3, 3)
        self.layer3_image_shape  = (batch_size, nkerns[1], 12, 10)
        self.layer3_filter_shape = (nkerns[2], nkerns[1], 3, 3)
        self.layer4_image_shape  = (batch_size, nkerns[2], 5, 4)
        self.layer4_filter_shape = (nkerns[3], nkerns[2], 2, 2)
        self.result_image_shape  = (batch_size, nkerns[3], 4, 3)

    def _build_layer_architecture(self, n_hidden, acti_func=relu):
        '''
        simple means the deepid layer input is only the layer4 output.
        layer1: convpool layer
        layer2: convpool layer
        layer3: convpool layer
        layer4: conv layer
        deepid: hidden layer
        '''
        x = T.matrix('x')

        print '\tbuilding the model ...'
        
        layer1_input = x.reshape(self.layer1_image_shape)
        self.layer1 = LeNetConvPoolLayer(self._rng,
                input        = layer1_input,
                image_shape  = self.layer1_image_shape,
                filter_shape = self.layer1_filter_shape,
                poolsize     = (2,2),
                W = self._exist_params[5][0],
                b = self._exist_params[5][1],
                activation   = acti_func)

        self.layer2 = LeNetConvPoolLayer(self._rng,
                input        = self.layer1.output,
                image_shape  = self.layer2_image_shape,
                filter_shape = self.layer2_filter_shape,
                poolsize     = (2,2),
                W = self._exist_params[4][0],
                b = self._exist_params[4][1],
                activation   = acti_func)

        self.layer3 = LeNetConvPoolLayer(self._rng,
                input        = self.layer2.output,
                image_shape  = self.layer3_image_shape,
                filter_shape = self.layer3_filter_shape,
                poolsize     = (2,2),
                W = self._exist_params[3][0],
                b = self._exist_params[3][1],
                activation   = acti_func)

        self.layer4 = LeNetConvLayer(self._rng,
                input        = self.layer3.output,
                image_shape  = self.layer4_image_shape,
                filter_shape = self.layer4_filter_shape,
                W = self._exist_params[2][0],
                b = self._exist_params[2][1],
                activation   = acti_func)

        layer3_output_flatten = self.layer3.output.flatten(2)
        layer4_output_flatten = self.layer4.output.flatten(2)
        deepid_input = T.concatenate([layer3_output_flatten, layer4_output_flatten], axis=1)

        self.deepid_layer = HiddenLayer(self._rng,
                input = deepid_input,
                n_in  = numpy.prod( self.result_image_shape[1:] ) + numpy.prod( self.layer4_image_shape[1:] ),
                n_out = n_hidden,
                W = self._exist_params[1][0],
                b = self._exist_params[1][1],
                activation = acti_func)
    
        self.generator = theano.function(inputs=[x],
                outputs=self.deepid_layer.output)


def deepid_generating(dataset_folder, params_file, result_folder, nkerns, n_hidden, acti_func=relu):
    if not dataset_folder.endswith('/'):
        dataset_folder += '/'
    if not result_folder.endswith('/'):
        result_folder += '/'
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    
    pd_helper = ParamDumpHelper(params_file)
    exist_params = pd_helper.get_params_from_file()
    if len(exist_params) != 0:
        exist_params = exist_params[-1]
    else:
        print 'error, no trained params'
        return
    
    dataset_files = os.listdir(dataset_folder)
    for dataset_file in dataset_files:
        dataset_path = dataset_folder + dataset_file
        result_path  = result_folder  + dataset_file
        x, y = load_data_xy(dataset_path)
        deepid = DeepIDGenerator(exist_params)
        deepid.layer_params(nkerns, x.shape[0])
        deepid.build_layer_architecture(n_hidden, acti_func)
        new_x = deepid.generate_deepid(x)
	cPickle_output((new_x, y), result_path)

def load_data_xy(dataset_path):
    print 'loading data of %s' % (dataset_path)
    f = open(dataset_path, 'rb')
    x, y = pickle.load(f)
    f.close()
    return x,y

def cPickle_output(vars, file_name):
    print '\twriting data to %s' % (file_name)
    import cPickle
    f = open(file_name, 'wb')
    cPickle.dump(vars, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'Usage: python %s dataset_folder params_file result_folder' % (sys.argv[0])
        sys.exit()

    dataset_folder = sys.argv[1]
    params_file    = sys.argv[2]
    result_folder  = sys.argv[3]
    nkerns  = [20,40,60,80]
    n_hidden = 160

    deepid_generating(dataset_folder, params_file, result_folder, nkerns, n_hidden, acti_func=relu)
