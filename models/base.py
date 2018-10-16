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
                self.namespace = 'base'        

            super(BaseModel, self).__init__(hparams)
            self.__helpers__()
            self.__placeholders__()
            self.__architecture__()

    def get_name(self):
        if self.contains('namespace'):
            return self.namespace.split('/')[-1]
        else:
            return 'base'

    def __helpers__(self):  
        self.layers = {}
        self.sess = None
        self.display_tensorboard = ['lr', '_weight_regularizer', 'wdecay', 'weight_penalty']
        with tf.variable_scope(self.namespace, reuse=tf.AUTO_REUSE):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
        

    def one_step(self):
        self.counter+=1
        increment_global_step_op = tf.assign(self.current_epoch, self.current_epoch+1)
        self.sess.run(increment_global_step_op)

    def __get_operation__(self, info, net=False):
        if info['op'] == 'placeholder':
            return self.get_param(info['input'])
        
        if info['op'] == 'concat':
            input_ = [self.get_param(inp) for inp in info['inputs']]
            if isinstance(net, bool) is False:
                input_ = [net, *input_]
            return __OPERATORS__[info['op']](values=input_, axis=info['axis'])
        
        if info['op'] == 'activation':
            if 'params' in info:
                return __OPERATORS__[info['type']](net, **info['params'])
            else:
                return __OPERATORS__[info['type']](net)

        if info['op'] == 'layers/activation':
            if not self.get_param(info['input']):
                with tf.variable_scope(self.namespace, reuse=tf.AUTO_REUSE):
                    if 'params' in info:
                        _layer = __INITIALIZERS__[info['type']](**info['params'])
                    else:
                        _layer = __INITIALIZERS__[info['type']]()
                    ans = _layer(net)
                    self.set_param(info['input'], _layer)
                    return ans
            else:
                return self.get_param(info['input'])(net)


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

    def set_loss(self, loss):
        self._loss = loss
        self.apply_wdecay()

        
    def add2loss(self, term):
        self._loss += term
        self.set_update()

    def loss(self):
        NotImplementedError("not implemented")

    def get_loss(self, data):
        _fd = self.get_feed_dict(data)
        return self.sess.run([self.loss()], feed_dict=_fd)[0]

    def apply_wdecay(self):
        with tf.variable_scope(self.namespace, reuse=tf.AUTO_REUSE):
            if 'wdecay' in self.__dict__:
                self.wdecay = tf.Variable(self.wdecay, dtype=tf.float32, trainable=False, name="wdecay")
            else:
                self.wdecay = tf.Variable(0, dtype=tf.float32, trainable=False)

            self._weight_regularizer = tf.constant(0.0, dtype=tf.float32)
        for _layer in self.get_layers():
            if '_fc' in _layer.name:
                self._weight_regularizer = tf.add(self._weight_regularizer, tf.nn.l2_loss(_layer))
        
        self.weight_penalty = tf.multiply(self.wdecay, self._weight_regularizer)
        self.add2loss(self.weight_penalty)

    def build(self, training=True):
        self.__build_specs__(training)
        self.__init_variables__()

    def __build_specs__(self, training=True):
        self.output = self.__forward__()
        self.output_test = self.__forward__(training=False)

        if training:
            self.set_loss(tf.constant(0, dtype=tf.float32))

    def update(self, data, tf_run=None):
        if tf_run is None:
            tf_run = [self._loss, self._update]
        return self.sess.run(tf_run, feed_dict=data)

    def get_feed_dict(self, data, idx=False):
        _fd = {}
        _placeholders = data['placeholders'] if 'placeholders' in data else self.placeholder_input
        for input_ in _placeholders:
            assert input_ in data
            if idx is False:
                _fd[self.get_param(input_)] = data[input_]
            else:
                _fd[self.get_param(input_)] = data[input_][idx]

        if self.contains('fix_variables'):
            for input_ in self.fix_variables:
                _fd[self.get_param(input_)] = data[input_]
        return _fd

    def train(self, data, batch_size=512):
        from numpy import array
        for input_ in self.placeholder_input:
            assert input_ in data
        
        _loss = array([])
        for ii_, xcur, xsize in self.next_batch(data, batch_size):

            ddata = self.get_feed_dict(data, ii_)
            out = self.update(ddata)
            _loss = self.merge_array(_loss, out[0])

            _msg = '{} [{}/{}] l: {:.3g}'.format(data['info'], xcur, xsize,
                                          _loss[-1])
            self.printer.print(_msg)

        try:
            _eval = self.evaluate(data)['{}_acc'.format(self.get_name())]
        except:
            _eval = -99.

        _msg = '{} [{}/{}] l: {:.3g} @t1: {:.3g} [resume]\n'.format(data['info'], xcur, xsize,
                                  _loss.mean(), _eval)
        self.printer(_msg)
        self.printer.next_line()


        self.one_step()
        self.params_tensorboard()


        return {'{}_loss'.format(self.get_name()): _loss.mean(),
                '{}_eval'.format(self.get_name()): _eval}

    def __call__(self, data, batch_size=512):
        from numpy import array
        
        for input_ in self.placeholder_call:
            assert input_ in data
        
        out = array([])
        for ii_, xcur, xsize in self.next_batch(data, batch_size, shuffle=False):
            fdict = {self.get_param(key): data[key][ii_] for key in self.placeholder_call}           
            out_ = self.sess.run([self.output_test], feed_dict=fdict)[0]
            out = self.merge_array(out, out_)

        return out

    def run(self, data, shuffle=False, batch_size=512): 
        from numpy import array

        for _key in (data['placeholders'] if data.get('placeholders') else self.placeholder_call):
            assert _key in data
        
        out = array([])
        for ii_, xcur, xsize in self.next_batch(data, batch_size, shuffle=shuffle):
            fdict = self.get_feed_dict(data, ii_)
            out_ = self.sess.run(data['output'], feed_dict=fdict)[0]
            out = self.merge_array(out, out_)

        return out

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
        self.__set_saver__()
        try:
            checkpoint_dir = data['dir']
            import re, os
            print(" [*] Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
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
        _path_split_ = save_path.split('/')
        if _path_split_ == '':
            _link = '{}/{}'.format('/'.join(_path_split_[:-2]), namelink)
        else:
            _link = '{}/{}'.format('/'.join(_path_split_[:-1]), namelink)
            
        try:
            remove(_link)
        except:
            pass
        symlink(save_path, _link)
    
    def __set_saver__(self):
        if not self.contains('saver'):
            self.saver = tf.train.Saver(var_list=tf.trainable_variables(scope=self.namespace))
            self.saver_full = tf.train.Saver()

    def save(self, data, save_all=False):
        assert 'dir' in data
        assert 'step' in data
        self.__set_saver__()

        self.saver.save(sess=self.sess,
                        save_path='{}/{}.model'.format(data['dir'], data['step']),
                        global_step=data['step'])
        if save_all:
            self.saver_full.save(sess=self.sess,
                    save_path='{}/{}.model'.format(data['dir'], data['step']),
                    global_step=data['step'])    
        self.__save_architecture__(data)
        self.link_checkpoint(data['dir'])



    def get_layers(self):
        return list(self.layers.values())

    def trainable_variables(self):
        return tf.trainable_variables(scope=self.namespace)

    def get_weights(self):
        return {key: self.sess.run([item])[0] for key, item in self.layers.items()}

    def __forward__(self, ret_all=False, training=True):
        operation = self.operators[0]
        net = self.__get_operation__(operation)
        return self.forward(net, ret_all=ret_all, training=training)

    def __set_operation_test__(self, operation):
        if operation['op']  == 'activation':
            if operation['type'] == 'dropout':
                if self.contains('test'):
                    if 'mcmc_dropout' in self.test:
                        return operation
                else:
                    operation['params']['keep_prob'] = 1.0
        
        return operation

    def forward(self, data, ret_all=False, training=True):
        net = data

        ret_layers = {}
        for operation in self.operators[1:]:
            if training == False:
                operation = self.__set_operation_test__(deepcopy(operation))
            
            net = self.__get_operation__(operation, net)
            if 'out' in operation and training:
                if operation['out']:
                    ret_layers[operation['ref']] = net

        if len(ret_layers) > 0 and ret_all:
            return net, ret_layers
        else:
            return net
    
    def accuracy(self, data):
        return self.accuracy_per_class(data['y_pred'], data['y'], self.y_dim)

    def __update_lr__(self):
        if 'current_epoch' not in self.__dict__:
            with tf.variable_scope(self.namespace, reuse=tf.AUTO_REUSE):
                self.current_epoch = tf.Variable(0, name='current_epoch', trainable=False, dtype=tf.int32)

        self.lr = tf.train.inverse_time_decay(learning_rate=self.optimizer['lr'] if 'lr' in self.optimizer else 1e-2,
                                              global_step=self.current_epoch,
                                              decay_rate=self.optimizer['decay'] if 'decay' in self.optimizer else 0,
                                              decay_steps=1)



    def set_update(self):
        self._update = self.update_step(self._loss)

    def params_tensorboard(self):
        if self.counter == 1:
            import json
            _arch = json.dumps(self.hyperparams, indent=4, sort_keys=False)
            self.set_summary(tag='{}/architecture'.format(self.namespace), 
                             value=_arch, stype='text')

        for _param in self.display_tensorboard:
            self.set_summary(tag='{}/params/{}'.format(self.namespace, _param), 
                             value=self.sess.run([self.get_param(_param)])[0])

    def update_step(self, loss, trainables=False):
        optimizer_func = self.__get_optimizer__(self.optimizer['name'] if 'name' in self.optimizer else 'adam')
        self.__update_lr__()

        with tf.variable_scope(self.namespace, reuse=tf.AUTO_REUSE):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                return optimizer_func(learning_rate=self.lr).minimize(loss, 
                                                                  var_list=trainables if trainables else self.trainable_variables(), 
                                                                  global_step=self.global_step,
                                                                  name='{}/optimizer'.format(self.namespace))

    def evaluate(self, data):
        NotImplementedError("not implmented yet")


    def predict(self, data):
        return self(data)

__MODEL__=BaseModel
