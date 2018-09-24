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

from .gan import GAN
from .classifier import Classifier
from copy import deepcopy

import numpy as np
class WGAN(GAN):

    def __init__(self, hparams):
        if 'namespace' not in hparams:
            hparams['namespace'] = 'wgan'
        
        if '__list_models__' not in hparams:
            hparams['__list_models__'] = ['generator', 'discriminator']
        super(WGAN, self).__init__(hparams)

    def cwgan_loss(self):
        alpha = tf.random_uniform(shape=tf.shape(self.generator.output), minval=0., maxval=1.)
        interpolation = alpha * self.generator.output + (1. - alpha) * self.generator.output
        
        d_input = tf.concat([interpolation, self.generator.a], -1)
        grad = tf.gradients(self.discriminator.forward(d_input), [interpolation])[0]
        grad_norm = tf.norm(grad, axis=1, ord='euclidean')
        self.grad_pen = self.lmbda * tf.reduce_mean(tf.square(grad_norm - 1))

        return self.d_real - (self.d_fake + self.aux_loss) + self.grad_pen

__MODEL__=WGAN
