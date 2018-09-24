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

from .base import ModelObject
from .discriminators import Discriminator
from .generators import Generator
from .classifier import Classifier
from .regressor import Regressor
from .base import BaseModel
from copy import deepcopy

import numpy as np

class GAN(ModelObject):

    def __init__(self, hparams):
        super(GAN, self).__init__(hparams)

        if hparams != None:
            if 'namespace' not in hparams:
                hparams['namespace'] = 'gan/'
        if '__list_models__' not in hparams:
          self.__list_models__ = ['generator', 'discriminator']
        self.__setup_models__(hparams)
        self.regularization = tf.Variable(-99, trainable=False)

    
    def get_models(self):
      return self.__list_models__

    def get_basic_model(self, value):
      return {'classifier': Classifier,
              'regressor': Regressor,
              'generator': Generator,
              'discriminator': Discriminator}[value]

    def __setup_models__(self, hparams):  
      for _model in self.__list_models__:
        hparams[_model]['placeholders'] = deepcopy(hparams['placeholders'])
        hparams[_model]['namespace'] = '{}/{}'.format(self.namespace, _model)
        self.__dict__[_model] = self.get_basic_model(_model)(hparams[_model])

    def __build_in__(self):
      for _model in self.__list_models__:
        self.__dict__[_model].build()

    def __build_specs__(self):
        # Generator step
        # self.g_loss = tf.reduce_mean(self.discriminator.forward(self.generator.output))
        self.generator._loss = tf.reduce_mean(self.discriminator.forward(self.generator.output))

        # Discriminator step
        self.d_real = tf.reduce_mean(self.discriminator.output)
        self.d_fake = tf.reduce_mean(self.discriminator.forward(self.generator.output))
        self.discriminator._loss = self.wgan_loss()

    def build(self):
        self.__build_in__()
        self.__build_specs__()

        self.generator._update = self.generator.update_step(self.generator._loss)
        self.discriminator._update = self.discriminator.update_step(self.discriminator._loss)

        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

    def get_update_variables(self, gtype='generator'):
      if gtype == 'generator':
        return [self.generator._loss, self.generator._update]
      elif gtype == 'discriminator':
        return [self.discriminator._loss, self.discriminator._update]
      else:
        return [self.__dict__[gtype]._loss, self.__dict__[gtype]._update]

    def update_generator(self, data):
        return self.sess.run([self.generator._loss, 
                              self.regularization,
                              self.generator._update], 
                              feed_dict={self.generator.a: data['a'],
                                         self.generator.z: data['z'],
                                         self.classifier.y: data['y']})

    def update_discriminator(self, data):
        return self.sess.run([self.discriminator._loss, 
                              self.d_real, 
                              self.d_fake, 
                              self.grad_pen, 
                              self.regularization,
                              self.discriminator._update], 
                              feed_dict={self.generator.a: data['a'],
                                         self.generator.z: data['z'],
                                         self.discriminator.a: data['a'],
                                         self.discriminator.x: data['x'],
                                         self.classifier.y: data['y']})

    def wgan_loss(self):
        alpha = tf.random_uniform(shape=tf.shape(self.generator.output), minval=0., maxval=1.)
        interpolation = alpha * self.generator.output + (1. - alpha) * self.generator.output
        
        grad = tf.gradients(self.discriminator.forward(interpolation), [interpolation])[0]
        grad_norm = tf.norm(grad, axis=1, ord='euclidean')
        self.grad_pen = self.lmbda * tf.reduce_mean(tf.square(grad_norm - 1))

        return self.d_real - self.d_fake + self.grad_pen

    def save(self, data):
      assert 'train_type' in data

      if data['train_type'] is 'gan':
        train_type = ['generator', 'discriminator']
      elif data['train_type'] is 'all':
        train_type = self.__list_models__
      else:
        train_type = list([data['train_type']])

      for _model in train_type:
          data['dir'] = '{}/{}/epoch_{}_{}'.format(data['checkdir'], 
                                                   _model, 
                                                   data['step'], 
                                                   data['epochs'])
          self.__dict__[_model].save(data)


    def set_session(self, sess):
        self.sess = sess
        for _model in self.__list_models__:
          self.__dict__[_model].set_session(sess)



    def train_gan(self, data, batch_size=512):
        for input_ in self.placeholder_input:
            assert input_ in data

        d_loss, g_loss = np.array([]), np.array([])
        d_real, d_fake, d_grad = np.array([]), np.array([]), np.array([])
        for ii_, xcur, xsize in self.next_batch(data, batch_size):
            
            gdata = {'a': data['a'][ii_], 
                     'z': self.get_noise(shape=data['a'][ii_].shape),
                     'y': data['y'][ii_]}
            
            ddata = {'a': data['a'][ii_], 
                     'z': self.get_noise(shape=data['a'][ii_].shape),
                     'x': data['x'][ii_],
                     'y': data['y'][ii_]}
            
            # Generator + Discriminator update
            out = self.update_discriminator(ddata)
            d_loss = self.merge_array(d_loss, np.array(out[0]))
            d_real = self.merge_array(d_real, np.array(out[1]))
            d_fake = self.merge_array(d_fake, np.array(out[2]))
            d_grad = self.merge_array(d_grad, np.array(out[3]))

            # Generator first update
            out = self.update_generator(gdata)
            g_loss = self.merge_array(g_loss, np.array(out[0]))


            _msg = '{} [{}/{}] g: {:.3g} | d: {:.3g}'.format(data['info'], xcur, xsize,
                                                       g_loss[-1], 
                                                       d_loss[-1])
            self.printer(_msg)
        

            _msg = '{} [{}/{}] g: {:.3g} | d: {:.3g} [resume]\n'.format(data['info'], xcur, xsize,
                                                 g_loss.mean(), 
                                                 d_loss.mean())
        self.printer(_msg)
        return {'discriminator_loss': d_loss.mean(),
                'discriminator_real': d_real.mean(),
                'discriminator_fake': d_fake.mean(),
                'discriminator_grad': d_grad.mean(),
                'generator_loss': g_loss.mean()}


    def train(self, data, batch_size=512):
        self.counter += 1
        
        if data['train_type'] is 'gan':
          return self.train_gan(data, batch_size=batch_size)
        else:
          return self.__dict__[data['train_type']].train(data, batch_size)

    def evaluate(self, data):
        if 'train_type' in data:
          if data['train_type'] is not 'gan':
            return self.__dict__[data['train_type']].evaluate(data)

        return self.classifier.evaluate(data)

__MODEL__ = GAN
