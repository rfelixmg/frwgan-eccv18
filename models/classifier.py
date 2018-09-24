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

class Classifier(BaseModel):

    def __init__(self, hparams=None):
        if hparams != None:
            if 'namespace' not in hparams:
                hparams['namespace'] = 'classifier'
        super(Classifier, self).__init__(hparams)
        self.loss_function  = self.bce_loss
        
    def loss(self):
        return self.bce_loss(self.outlogit, self.y)

    def __build_specs__(self):
        self.output, out = self.__forward__(ret_all=True)
        self.outlogit = out['last_logit']

        self._loss = self.loss()
        self._update = self.update_step(self._loss)
        self.c_accuracy = tf.metrics.mean_per_class_accuracy(self.y,
                                                             self.y_pred,
                                                             self.y_dim)
    def bce_loss(self, y_pred, y_true):
        out = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        return tf.reduce_mean(out)

    def evaluate(self, data):
        y_pred = self(data)
        return self.accuracy({'y_pred': y_pred.argmax(-1), 
                              'y': data['y'].argmax(-1)})


__MODEL__=Classifier
