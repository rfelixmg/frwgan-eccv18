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
from .modelobj import ModelObject
from copy import deepcopy
from .supporttypes import __OPERATORS__, __INITIALIZERS__, __OPTIMIZERS__

class BaseModel(ModelObject):

    def __init__(self, hparams=None):
        if hparams is not None:
            self.hyperparams = deepcopy(hparams)
            if 'namespace' not in hparams:
                self.namespace = 'base/'        

            super(BaseModel, self).__init__(hparams)
            self.__helpers__()
            self.__placeholders__()
            self.__architecture__()

    def __helpers__(self):  
        self.layers = {}
        self.sess = None
        self.saver = None
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.lr = tf.Variable(0, trainable=False, name='learning_rate')

    def __get_operation__(self, info, net=False):
        if info['op'] == 'placeholder':
            return self.get_param(info['input'])
        
        if info['op'] == 'concat':
            input_ = [self.get_param(inp) for inp in info['inputs']]
            if isinstance(net, bool) is False:
                input_ = [net, *input_]
            return __OPERATORS__[info['op']](values=input_, axis=info['axis'])
        
        if info['op'] == 'activation':
            return __OPERATORS__[info['type']](net)

        elif isinstance(net, bool) is False:
            return __OPERATORS__[info['op']](net, self.get_param(info['input']))
        
        else:
            input_ = [self.get_param(inp) for inp in info['inputs']]
            return __OPERATORS__[info['op']](*input_)


    def __get_initializer__(self, info):
        if 'params' in info:
            return __INITIALIZERS__[info['name']](**info['params'])
        else:
            return __INITIALIZERS__[info['name']]()


    def __placeholders__(self):
        assert 'placeholders' in self.__dict__
        with tf.variable_scope(self.namespace):

            for ph in self.placeholders:
                ph['dtype'] = tf.__dict__[ph['dtype']]
                self.__dict__[ph['name']] = tf.placeholder(**ph)

    def __get_optimizer__(self, name):
        return __OPTIMIZERS__[name]



    def __architecture__(self):
        with tf.variable_scope(self.namespace):
            for layer in self.architecture:
                layer['initializer'] = self.__get_initializer__(layer['initializer'])
                self.layers[layer['name']] = tf.get_variable(**layer)

    def get_placeholders(self):
        assert 'placeholders' in self.__dict__
        return [self.__dict__[ph['name']] for ph in self.placeholders]


    def get_architecture(self):
    	return self.layers

    def loss(self):
        NotImplementedError("not implemented")

    def __build_specs__(self):
        self.output = self.__forward__()
        self._loss = self.loss()

    def build(self):   
        self.__build_specs__()
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

    def update(self, data, tf_run=None):
        if tf_run is None:
            tf_run = [self._loss, self._update]
        
        _fd = {}
        for input_ in self.placeholder_input:
            assert input_ in data
            _fd[self.get_param(input_)] = data[input_]
        return self.sess.run(tf_run, feed_dict=_fd)

    def build(self):
        self.__build_specs__()
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
            
    def train(self, data, batch_size=512):
        from numpy import array
        for input_ in self.placeholder_input:
            assert input_ in data
        
        _loss = array([])
        for ii_, xcur, xsize in self.next_batch(data, batch_size):
            
            ddata = {}
            for input_ in self.placeholder_input:
                 ddata[input_] = data[input_][ii_]
            
            out = self.update(ddata)
            _loss = self.merge_array(_loss, out[0])

            _msg = '{} [{}/{}] l: {:.3g}'.format(data['info'], xcur, xsize,
                                          _loss[-1])
            self.printer(_msg)

        try:
            _eval = self.evaluate(data)['acc']
        except:
            _eval = -99.

        _msg = '{} [{}/{}] l: {:.3g} @t1: {:.3g} [resume]\n'.format(data['info'], xcur, xsize,
                                  _loss.mean(), _eval)
        self.printer(_msg)

        return {'{}_loss'.format(self.namespace): _loss.mean(),
                'eval': _eval}

    def __call__(self, data, batch_size=512):
        from numpy import array
        
        for input_ in self.placeholder_call:
            assert input_ in data
        
        out = array([])
        for ii_, xcur, xsize in self.next_batch(data, batch_size, shuffle=False):
            fdict = {self.get_param(key): data[key][ii_] for key in self.placeholder_call}           
            out_ = self.sess.run([self.output], feed_dict=fdict)[0]
            out = self.merge_array(out, out_)

        return out


    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.namespace in var.name]


    def __save_architecture__(self, data):
        import json
        assert 'dir' in data        
        with open('{}/architecture.json'.format(data['dir']), 'w') as outfile:
            json.dump(self.hyperparams, outfile)


    @classmethod
    def load_architecture(self, data):        
        """
        :param: @data: {'model':obj(BaseModel), 'architecture': txt(file_path)}
        :return: obj(BaseModel)
        """
        import json
        with open(data['architecture']) as f:
            hparams = json.load(f)
        data['model'].__init__(hparams)

    def load(self, data):
        if self.saver is None:
            self.__set_saver__()
        try:
            checkpoint_dir = data['dir']
            import re, os
            print(" [*] Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            print(ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
                print(" [*] Success to read {}".format(ckpt_name))
                return True, counter
            else:
                print(" [*] Failed to find a checkpoint")
                return False, 0
        except:
            raise

    def link_checkpoint(self, save_path, namelink='last_epoch'):
        from os import remove, symlink

        _link = '{}/{}'.format('/'.join(save_path.split('/')[:-1]), namelink)
        try:
            remove(_link)
        except:
            pass
        symlink(save_path, _link)

    def save(self, data):
        assert 'dir' in data
        assert 'step' in data

        if self.saver is None:
            self.__set_saver__()

        self.saver.save(sess=self.sess,
                        save_path='{}/{}.model'.format(data['dir'], data['step']),
                        global_step=data['step'])
        self.__save_architecture__(data)
        self.link_checkpoint(data['dir'])


    def __set_saver__(self):
        self.saver = tf.train.Saver()

    def set_session(self, sess):
        self.sess = sess

    def get_session(self, sess):
        return self.sess


    def get_trainable(self):
        return list(self.layers.values())

    def get_weights(self):
        return {key: self.sess.run([item])[0] for key, item in self.layers.items()}

    def __forward__(self, ret_all=False):
        operation = self.operators[0]
        net = self.__get_operation__(operation)
        return self.forward(net, ret_all=ret_all)

    def forward(self, data, ret_all=False):
        net = data

        ret_layers = {}
        for operation in self.operators[1:]:
            net = self.__get_operation__(operation, net)
            if 'out' in operation:
                if operation['out']:
                    ret_layers[operation['ref']] = net

        if len(ret_layers) > 0 and ret_all:
            return net, ret_layers
        else:
            return net
    
    def accuracy(self, data):
        return {'acc': self.accuracy_per_class(data['y_pred'], data['y'], self.y_dim)}

    def __update_lr__(self):
        self.lr = tf.train.inverse_time_decay(learning_rate=self.optimizer['lr'] if 'lr' in self.optimizer else 1e-2,
                                         global_step=self.global_step,
                                         decay_rate=self.optimizer['decay'] if 'decay' in self.optimizer else 0,
                                         decay_steps=1)

    def update_step(self, loss):
        optimizer_func = self.__get_optimizer__(self.optimizer['name'] if 'name' in self.optimizer else 'adam')
        self.__update_lr__()
        return optimizer_func(learning_rate=self.lr).minimize(loss, 
                                                              var_list=self.get_trainable(), 
                                                              global_step=self.global_step)

    def evaluate(self, data):
        NotImplementedError("not implmented yet")


__MODEL__=BaseModel
