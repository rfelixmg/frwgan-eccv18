"""
MIT License

Copyright (c) 2018 Rafael Felix Alves

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from .base import BaseModel
import tensorflow as tf

class Regressor(BaseModel):

    def __init__(self, hparams=None):
        if hparams != None:
            if 'namespace' not in hparams:
                hparams['namespace'] = 'regressor'
        super(Regressor, self).__init__(hparams)
        self.loss_function  = self.mse_loss
    
    def mse_loss(self, a_pred, a_true):
        return tf.reduce_mean(tf.squared_difference(a_pred, a_true))

    def loss(self):
        return self.mse_loss(self.output, self.a)

    def __build_specs__(self, training=True):
        self.output = self.__forward__()
        self.output_test = self.__forward__(training=False)
        self.distances = tf.reduce_sum(tf.abs(tf.subtract(self.a_dict, 
                                                          tf.expand_dims(self.a_pred,1))), axis=2)
        self.knn = tf.argmin(self.distances, 1)
        
        if self.contains('test'):
            if 'mcmc_dropout' in self.test:
                self.output_mcmc = self.__forward__()
        if training:
            self.set_loss(self.loss())

    def get_knn(self, a_pred, a_dict):
        return self.sess.run([self.knn], 
                             feed_dict={self.a_pred: a_pred, self.a_dict: a_dict})

    def get_distances(self, data):
        tdata = {'output':[self.distances], 
                 'placeholder_batch':'a_pred',
                 'placeholders':['a_pred']}
        tdata.update(data)
        return self.run(tdata)

    def get_y_pred(self, a_pred, data):
        import numpy as np
        a_masked = np.ma.array(a_pred, mask=np.array([data['y_classes']] * a_pred.shape[0]))
        if 'op' in data:
            if data['op'] == 'min':
                return a_masked.argmin(-1)
            if data['op'] == 'max':
                return a_masked.argmax(-1)
        else:
            return a_masked.argmin(-1)

    def evaluate(self, data):
        a_pred = self(data)
        data['a_pred'] = a_pred
        y_prob = self.get_distances(data)
        
        y_pred = self.get_y_pred(y_prob, data)
        return {'{}_loss'.format(self.get_name()) : self.get_loss(data),
                '{}_acc'.format(self.get_name()): self.accuracy({'y_pred': y_pred,  'y': data['y'].argmax(-1)})}

    def predict(self, data):
        from numpy import array
        from util.metrics import softmax
        _data = {'a_pred': self(data)}
        _data.update(data)
        return softmax(-self.get_distances(_data), -1)


__MODEL__=Regressor
