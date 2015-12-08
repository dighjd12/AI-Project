#!/usr/bin/env python

import numpy as np
import cv2

#todo: 1. reduce the size of input features - right now 32^2*3 long..
# 2. make it such that after training, it can take photos and return the prediction result
# 4. play around with parameters ?
# 5. also, right now only using one of the 5 dataset. should we change this?

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

class LetterStatModel(object):
    #number of possible outputs.
    class_n = 2
    #train ratio of the dataset
    train_ratio = 0.5

    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

    def unroll_samples(self, samples):
        sample_n, var_n = samples.shape
        new_samples = np.zeros((sample_n * self.class_n, var_n+1), np.float32)
        new_samples[:,:-1] = np.repeat(samples, self.class_n, axis=0)
        new_samples[:,-1] = np.tile(np.arange(self.class_n), sample_n)
        return new_samples

    def unroll_responses(self, responses):
	#print(responses)
        sample_n = len(responses)
        new_responses = np.zeros(sample_n*self.class_n, np.int32)
        resp_idx = np.int32( responses + np.arange(sample_n)*self.class_n )
        #print resp_idx
        #print sample_n
	new_responses[resp_idx] = 1
	#print(new_responses)
        return new_responses

class MLP(LetterStatModel):
    def __init__(self):
        self.model = cv2.ANN_MLP()
        self.train_ratio = 0.8

    def train(self, samples, responses):
        print("START TRAINING")
        sample_n, var_n = samples.shape
        new_responses = self.unroll_responses(responses).reshape(-1, self.class_n)

        layer_sizes = np.int32([var_n, 100, self.class_n])
        self.model.create(layer_sizes)

        # CvANN_MLP_TrainParams::BACKPROP,0.001
        params = dict( term_crit = (cv2.TERM_CRITERIA_COUNT, 300, 0.01),
                       train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,
                       bp_dw_scale = 0.1,
                       bp_moment_scale = 0.1 )
        self.model.train(samples, np.float32(new_responses), None, params = params)
        print("END TRAINING")

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)

def getData():
    data_list = [
        unpickle('data/cifar-10-batches-py/data_batch_1'),
        unpickle('data/cifar-10-batches-py/data_batch_2'),
        unpickle('data/cifar-10-batches-py/data_batch_3'),
        unpickle('data/cifar-10-batches-py/data_batch_4'),
        unpickle('data/cifar-10-batches-py/data_batch_5'),
        unpickle('data/cifar-10-batches-py/test_batch'),
    ]

    #samples = np.empty_like(data_list[0]['data'].astype(np.float32))
    samples = data_list[0]['data'].astype(np.float32)
    labels = np.array(data_list[0]['labels'], dtype=np.float32)
    for idx, d in enumerate(data_list):
        if idx != 0:
            samples = np.concatenate([samples, d['data'].astype(np.float32)])
            labels = np.concatenate([labels, np.array(d['labels'], dtype=np.float32)])

    img_idxs = []
    for idx, l in enumerate(labels):
    	# 3 is cat, 5 is dog
    	if l == 3 or l == 5:
            img_idxs.append(idx)

    data = samples[img_idxs]
    responses = labels[img_idxs]
    print("Number of images used: " + str(len(img_idxs)))

    for idx, l in enumerate(responses):
        if l == 3:
            responses[idx] = 0
        else:
            responses[idx] = 1

    return (data, responses)

if __name__ == '__main__':
    import getopt
    import sys

    # samples = attributes, responses = the answer
    (samples, responses) = getData()

    model = MLP()

	# train_n: number of training samples
    print("Train ratio: {}".format(model.train_ratio))
    train_n = int(len(samples)*model.train_ratio)

    model.train(samples[:train_n], responses[:train_n])

    print 'testing...'
    train_rate = np.mean(model.predict(samples[:train_n]) == responses[:train_n])
    test_rate  = np.mean(model.predict(samples[train_n:]) == responses[train_n:])

    print 'train rate: %f  test rate: %f' % (train_rate*100, test_rate*100)
