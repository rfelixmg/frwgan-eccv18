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

train.py: Training routine.
    Reference: 
            Felix, R., Vijay Kumar, B. G., Reid, I., & Carneiro, G. (2018). Multi-modal Cycle-consistent Generalized Zero-Shot Learning. arXiv preprint arXiv:1808.00136.
            (download paper)[http://openaccess.thecvf.com/content_ECCV_2018/papers/RAFAEL_FELIX_Multi-modal_Cycle-consistent_Generalized_ECCV_2018_paper.pdf]
"""

__author__ = "Rafael Felix, Vijay Kumar and Gustavo Carneiro"
__copyright__ = "MIT License"
__credits__ = ["Rafael Felix", "Gustavo Carneiro", "Vijay Kumar", "Ian Reid"]
__license__ = "MIT License"
__version__ = "1.0.2"
__maintainer__ = "Rafael Felix"
__email__ = "rafael.felixalves@adelaide.edu.au"
__status__ = "production"



_repeat_=100
from .aux import __tensorboard_script__, __seed__, __git_version__, __add_parent__, update_metric



def initialize(params):
    from util import setup, storage
    from os import environ

    if params.opt.setup:
        # Generating experimental setup folder
        print(':: Generating experimental setup folder')
        res = setup.mkexp(baseroot=params.opt.baseroot,
                          options=params,
                          bname='{}_{}'.format(params.opt.description, params.opt.timestamp),
                          sideinfo=params.opt.sideinfo,
                          subdirectories=params.opt.exp_directories)
        params.opt.root = res['root']

        for key in params.opt.exp_directories:
            params.opt.__setattr__('{}dir'.format(key), '{}/{}/'.format(params.opt.root, key))

        params.opt.namespace = res['namespace']
        print(':: Experiment will be save in:\n:: {}'.format(params.opt.root))
        options.checkpointdir = '{}/checkpoint/'.format(params.opt.root)
    params.save('{}/configuration_{}.json'.format(params.opt.root, params.opt.timestamp))
    params.print()
    params.opt.architecture = storage.Container(storage.Json.load(options.architecture_file))
    if params.opt.gpu_devices:
        environ["CUDA_VISIBLE_DEVICES"] = params.opt.gpu_devices

    try:
        version = __git_version__()
        storage.Json().save(version, '{}/git_version.json'.format(params.opt.root, params.opt.timestamp))
    except Exception as e:
        print(':: Exception:: {}'.format(e))
        print(Warning(':: Warning:: This project is not versioned yet'))

def augment_dataset():
    from util.tensors import merge_dict
    if options.augm_file:
        print(":: Augmenting original dataset")
        datafake = datasets.load_h5(options.augm_file)
        if options.augm_operation == 'merge':
            print(":: Merging augmented dataset to original dataset")
            dataset.train = Container(merge_dict(dataset.train.as_dict(), datafake.train.as_dict()))
        elif options.augm_operation == 'replace':
            print(":: Replacing original dataset by augmented dataset")
            dataset.train = datafake.train
        else:
            from warnings import warn
            warn(':: [warning] [default=merge] Augmenting operation not selected!')
            dataset.train = Container(merge_dict(dataset.train.as_dict(), datafake.train.as_dict()))
    return dataset


def train(model, params, dataset, knn, options, info=''):
    from util.experiments import label2hot, generate_metric_list
    from util.storage import Json, Dict_Average_Meter
    from util.metrics import accuracy_per_class
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    response = Dict_Average_Meter()
    epochs = params.epochs if 'epochs' in params.__dict__.keys() else 10
    batch_size = params.batch if 'batch' in params.__dict__.keys() else 512

    # Splitting dataset into validation and 
    print('='*50, '\n:: [{}]Initializing training...'.format(model.namespace))
    _split = train_test_split(dataset.train.X,
                              dataset.train.Y-1,
                              dataset.train.A.continuous,
                              test_size=options.validation_split,
                              random_state=42)
    X_train, X_val, y_train, y_val, a_train, a_val = _split

    if options.domain is 'openval':
        y_classes = np.zeros(knn.openset.data.shape[0])
        y_classes = y_classes[knn.zsl.ids-1] = 1.
    elif options.domain is 'zsl':
        y_classes = np.zeros(knn.openset.data.shape[0])
        y_classes[knn.openval.ids-1] = 1.
    else:
        y_classes = np.zeros(knn.openset.data.shape[0])
    y_evals = np.zeros(knn.openset.data.shape[0])

    # Iteration over epochs
    for epoch in np.arange(1, epochs+1):
        data = {'x': X_train,
                'y': label2hot(y_train, dataset.n_classes),
                'a': a_train,
                'a_dict': knn.openset.data.astype(np.float32),
                'y_classes': y_classes.astype(np.float32),
                'info':':: || {}[{}] - Epochs {}/{} ||'.format(info, model.namespace, epoch, epochs),
                'train_type':model.namespace}
        
        train_answer = model.train(data, batch_size=batch_size)
        train_eval = model.evaluate(data)

        response.update_meters("{}/train/answer".format(model.namespace), train_answer)
        model.summary_dict("{}/train/answer".format(model.namespace), train_answer)

        response.update_meters("{}/train/val".format(model.namespace), train_eval)
        model.summary_dict("{}/train/val".format(model.namespace), train_eval)

        if options.validation_split:
            val = {'x': X_val,
                   'y': label2hot(y_val, dataset.n_classes),
                   'a_dict': knn.openset.data,
                   'y_classes': y_classes.astype(np.float32),
                   'a': a_val,
                   'train_type':model.namespace}
            val_answer = model.evaluate(val)
            response.update_meters("{}/val".format(model.namespace), val_answer)
            model.summary_dict("{}/val".format(model.namespace), val_answer)

        if ((options.save_model) and (epoch >= options.save_from)) or \
        (epoch in options.savepoints):
            model.save({'dir': '{}/epoch_{}_{}'.format(options.checkpointdir, epoch, epochs),
                        'step':epoch})

    return response.as_dict()


def main(options, dataset, knn):

    from util.storage import DataH5py, Json
    from util.setup import mkdir
    from routines.aux import get_basic_model
    import models
    
    _archfile = Json.load(options.architecture_file)
    if options.load_model:
        from .aux import load_model
        model = load_model(options.load_model, _archfile.namespace)
    else:
        # Setting model from json file architecture
        print(':: Creating new model. ')
        ModelClass = models.__dict__[_archfile['namespace']].__MODEL__
        model = ModelClass(_archfile)
        print(":: Model type:", type(model))
        
        # Setting session
        print(':: Setting TensorFlow session. ')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=options.gpu_memory)
        config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        sess = tf.Session(config=config)
        model.set_session(sess)
        model.build()
    try:
        model.set_writer('{}/{}/'.format(options.logsdir, model.namespace))
        response = train(model=model, 
                         params=options.architecture,
                         dataset=dataset, knn=knn,
                         options=options,
                         info='{}::[{}/{}]: '.format(model.namespace, options.domain, options.dbname))
        print("")
        DataH5py().save_dict_to_hdf5(dic=response, filename='{}/full_train.h5'.format(options.resultsdir))

    except:
        import sys, traceback
        traceback.print_exc(file=sys.stdout)

    return model, None




if __name__ == '__main__':
    print('-'*100)
    print(':: Training file: {}'.format(__file__))
    print('-'*100)
    
    from options.benchmark import __OPTION__  as Options
    from util import datasets
    from util.storage import Container
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    from models import *
    
    try:
        print(':: Seeding to guarantee reproducibility')
        __seed__()
        print(':: Parsing parameters')
        params = Options()
        options = params.parse()

        print('-'*_repeat_,'\n:: Initializing experiment')
        initialize(params)
        print('-'*_repeat_,'\n:: Loading Dataset')
        dataset, knn = datasets.load(options.datadir)
        if options.domain == 'openset':
            dataset = augment_dataset()

        print('-'*_repeat_, '\n:: Generating tensorboard script')
        _tbscript_file_='tensorboard_script.sh'
        __tensorboard_script__(fname='/tmp/{}'.format(_tbscript_file_),
                               logidr=options.root)
        
        __tensorboard_script__(fname='{}/{}'.format(options.root, _tbscript_file_),
                               logidr=options.root)
        print(':: tensorboard script: {}'.format(_tbscript_file_))

        print('-'*_repeat_, '\n:: Executing main routines\n', '-'*_repeat_)

        model, results = main(options=options, dataset=dataset, knn=knn)
        print('-'*_repeat_,"\n Ending execution...\n", '-'*_repeat_,)
        print(':: Logs:\n', options.root, '\n','-'*_repeat_)
        
    except Exception as e:
        import sys, traceback
        traceback.print_exc(file=sys.stdout)