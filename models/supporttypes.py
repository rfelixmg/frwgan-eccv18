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
__INITIALIZERS__ = {'truncated': tf.truncated_normal_initializer,
                    'constant': tf.constant_initializer,
                    'zeros': tf.zeros_initializer,
                    'xavier': tf.contrib.layers.xavier_initializer}

__OPERATORS__ = {'matmul': tf.matmul,
                 'bias_add': tf.nn.bias_add,
                 'concat': tf.concat,
                 'relu': tf.nn.relu,
                 'softmax': tf.nn.softmax,
                 'leaky_relu': tf.nn.leaky_relu,
                 'sigmoid': tf.nn.sigmoid}

__OPTIMIZERS__ = {'adam': tf.train.AdamOptimizer,
                  'sgd': tf.train.GradientDescentOptimizer,
                  'rms': tf.train.RMSPropOptimizer}