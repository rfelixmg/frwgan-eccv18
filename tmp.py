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
import numpy as np
from models import *
import models
print(models.__dict__['base'])

model = models.base.__MODEL__()
_model=['regressor',
		'classifier', 
		'generator', 
		'discriminator']
load_model = '/tmp/test/0000_TEST_240918_183557/checkpoint/{}/last_epoch/'.format(_model[3])
pkg= {'model':model, 
	  'architecture': '{}/architecture.json'.format(load_model),
	  'dir':load_model}
models.base.__MODEL__.load_architecture(pkg)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
sess = tf.Session(config=config)
model.set_session(sess)
model.build()
model.load(pkg)

test_data = {'x': np.random.rand(100, 2048), 
			 'a': np.random.rand(100, 85),
			 'z': np.random.rand(100, 85)}
out = model(test_data)
print(out, '\n', out.shape)