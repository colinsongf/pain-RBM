"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""
import time

try:
    import PIL.Image as Image
except ImportError:
    import Image

import sys
import numpy as np
import scipy.io as sio

import theano
import theano.tensor as T
import os

from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images



# start-snippet-1
class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """
    def __init__(
        self,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        hbias=None,
        vbias=None,
        numpy_rng=None,
        theano_rng=None
    ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            # create a number generator
            numpy_rng = np.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]
        # end-snippet-1

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term


    def propup(self, vis):
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    # start-snippet-2
    def get_cost_updates(self, lr=0.02, persistent=None, k=1):

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k
        )
        # start-snippet-3
        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        # end-snippet-3 start-snippet-4
        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            )
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates
        # end-snippet-4


    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):

        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy


def test_rbm(learning_rate=0.01, training_epochs = 25, current_ID = 7137,
             batch_size=20,
             n_chains=20, n_samples=3, output_folder='rbm_plots',
             n_hidden=20):
    data_filename = '../train_data_' + str(current_ID) + '_test.mat';
    print data_filename
    train_set_x, test_set_x = load_data(data_filename)

    print 'Training set size is: ', train_set_x.get_value(borrow=True).shape
    print 'Test set size: ', test_set_x.get_value(borrow=True).shape
    dimensionData = test_set_x.get_value(borrow=True).shape[1]
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(np.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    # construct the RBM class
    rbm = RBM(input=x, n_visible= dimensionData,
              n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
 
    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain, k=15)

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # start-snippet-5
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    plotting_time = 0.
    start_time = time.clock()
    showeightsimage = False
    # go through training epochs
    for epoch in xrange(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        # print 'Training epoch %d, cost is ' % epoch, np.mean(mean_cost)

        # Plot filters after each training epoch
        plotting_start = time.clock()
        # Construct image from the weight matrix
	'''
	Show more information about the trained parameters
	print 'Weight shape is: ', rbm.W.get_value(borrow=True).T.shape
	print 'visible bias is: ', rbm.vbias.get_value(borrow=True).T
	print 'hidden bias is: ', rbm.hbias.get_value(borrow=True).T
        '''
	if showeightsimage:
		image = Image.fromarray(
        	    tile_raster_images(
                	X=rbm.W.get_value(borrow=True).T,
	                img_shape=(2, 11),
        	        tile_shape=(10, 10),
                	tile_spacing=(1, 1)
            		)
        	)
        	image.save('filters_at_epoch_%i.png' % epoch)

        plotting_stop = time.clock()
        plotting_time += (plotting_stop - plotting_start)

    end_time = time.clock()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))
    # end-snippet-5 start-snippet-6
    ######################################################################
    # Classification using free energy
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]
    print "The number of samples is: ", number_of_test_samples 
    TP, TN, FP, FN = [0, 0, 0, 0]
    idx_range = xrange(number_of_test_samples)
    xx = T.matrix()
    fe_fn = theano.function([xx], rbm.free_energy(T.round(xx)))
    record_num = 80
    RBMroc = np.ones((record_num, 2))	# Matrix used to store the value of TPR and FPR 
    predict_choice = 18		# Modify and test here 
    RBM_record_labels = np.ones(number_of_test_samples)
    count = 0
    for multiplier in xrange(1,record_num + 1,1):
	count += 1
	TP, TN, FP, FN = [0, 0, 0, 0]
	real_multiplier = multiplier / 20.0 ;	# the step parameter can be set any integer number, no big difference since the number of test is not huge
   	for predict_idx in idx_range:
		test = test_set_x.get_value(borrow = True)[predict_idx:predict_idx+1]
		real_label = test[0, dimensionData - 1]	# record the real label
		test[0, dimensionData - 1] = 0
		test1 = test_set_x.get_value(borrow = True)[predict_idx:predict_idx+1]
        	test1[0, dimensionData - 1] = 1
    		feLabel0 = fe_fn(test)	# free energy of label 0
		feLabel1 = fe_fn(test1)	# free energy of label 1

		if np.exp(-feLabel0) >= real_multiplier * np.exp(-feLabel1):
			if multiplier == predict_choice:
				RBM_record_labels[predict_idx] = 0	
			if real_label <= 0.5:
				TN += 1
			else:
				FN += 1	
		else:
  			if real_label < 0.5:
				FP += 1
			else:
				TP += 1

	TPR = (float(TP) / (TP + FN))
	FPR = (float(FP) / (FP + TN))
	RBMroc[multiplier-1, :] = [FPR, TPR]
	print 'TP, FP, TN, FN, FPR, TPR, count', TP, FP, TN, FN, FPR, TPR, count
    print RBM_record_labels
    area = rocarea(RBMroc)
    os.chdir('..')
  
    current_folder = str(current_ID)
    if not os.path.exists(current_folder):
        os.makedirs(current_folder)
    
    sio.savemat((current_folder + '/RBMroc.mat'), mdict = {'RBMroc':RBMroc, 'RBM_record_labels':RBM_record_labels})	# make sure saved in current repository
    return area

def load_data(filename):
	mat_dict = sio.loadmat(filename, struct_as_record = False, squeeze_me = True)
	train_Patient = mat_dict['train_data']	# the name of variable in matlab
	test_Patient = mat_dict['test_data']
	shared_train_Patient = theano.shared(np.asarray(train_Patient, dtype = theano.config.floatX))
	shared_test_Patient = theano.shared(np.asarray(test_Patient, dtype = theano.config.floatX))
	return shared_train_Patient, shared_test_Patient
	
def rocarea(roc):
	if roc.shape[0] == 0:
		sys.exit("Errors! There is no ROC matrix")
	if np.max(roc) > 1 or np.min(roc) < 0:
		sys.exit("Errors! There are unreasonable values in ROC matrix!")			
	if np.max(roc[:,0] < 1):
		roc = np.vstack([roc, [1,1]])
	if np.min(roc[:,0] > 0):
		roc = np.vstack([roc, [0,0]])

	indx = np.lexsort((roc[:,0], roc[:,1]))
	sorted_roc = roc[indx,:]
	area = 0
	min_dist = 1
	for i in range(len(roc[:,0])):
		if i == 0:
			area = area + sorted_roc[i, 1] * sorted_roc[i, 0] / 2.
		else:
			area = area + (sorted_roc[i, 1] + sorted_roc[i - 1, 1]) * (sorted_roc[i, 0] - sorted_roc[i-1, 0]) / 2.
		dist = sorted_roc[i, 0] * sorted_roc[i, 0] + (1-sorted_roc[i, 1]) * (1-sorted_roc[i, 1])
		if dist < min_dist:
			min_dist = dist
	print min_dist
	return area	

if __name__ == '__main__':
	area = test_rbm(learning_rate = 0.55, n_hidden = 33, current_ID = 6563) 
	print area
		

