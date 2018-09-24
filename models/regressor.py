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
        self.weight_regularizer = 0.
        for key, _layer in self.layers.items():
            self.weight_regularizer += self.wdecay * tf.nn.l2_loss(_layer)
        # self.weight_regularizer = self.wdecay * tf.nn.l2_loss(self.layers['r_fc1'])
        return tf.reduce_mean(tf.squared_difference(a_pred, a_true) + self.weight_regularizer)

    def loss(self):
        return self.mse_loss(self.output, self.a)

    def __build_specs__(self):
        self.output = self.__forward__()
        self._loss = self.loss()
        self._update = self.update_step(self._loss)  
        self.knn = tf.argmin(tf.reduce_sum(tf.abs(tf.subtract(self.a_dict, tf.expand_dims(self.a_pred,1))), 
                                axis=2), 1)

    def get_knn(self, a_pred, a_dict):
        return self.sess.run([self.knn], feed_dict={self.a_pred: a_pred, self.a_dict: a_dict})

    def evaluate(self, data):
        a_pred = self(data)
        y_pred = self.get_knn(a_pred, data['dict'])[0]
        return self.accuracy({'y_pred': y_pred,  'y': data['y'].argmax(-1)})




__MODEL__=Regressor
