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
import tensorflow as tf
from .wgan import WGAN
from .classifier import Classifier
from copy import deepcopy

import numpy as np

class RWGAN(WGAN):

    def __init__(self, hparams):
        if hparams != None:
            if 'namespace' not in hparams:
                hparams['namespace'] = 'rwgan'
        if '__list_models__' not in hparams:
            hparams['__list_models__'] = ['regressor', 'classifier', 'generator', 'discriminator']
        super(RWGAN, self).__init__(hparams)

    def __build_specs__(self):

        # Generator step
        d_input = tf.concat([self.generator.output, self.generator.a], -1)
        # self.g_loss = tf.reduce_mean(self.discriminator.forward(d_input))
        self.generator._loss = tf.reduce_mean(self.discriminator.forward(d_input))

        a_out = self.regressor.forward(self.generator.output, ret_all=True)[0]
        y_logit = self.classifier.forward(self.generator.output, ret_all=True)[1]['last_logit']

        self.regularization = self.regressor.loss_function(a_out, self.generator.a)
        self.aux_loss = self.regularization * self.regressor.beta
        
        self.generator._loss += self.aux_loss
        self.generator._update = self.generator.update_step(self.generator._loss)

        # Discriminator step
        self.d_real = tf.reduce_mean(self.discriminator.output)
        d_input_fake = tf.concat([self.generator.output, self.discriminator.a], -1)
        self.d_fake = tf.reduce_mean(self.discriminator.forward(d_input_fake))

        self.discriminator._loss = self.cwgan_loss()
        self.discriminator._update = self.discriminator.update_step(self.discriminator._loss)

__MODEL__=RWGAN
