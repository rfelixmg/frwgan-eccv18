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

    params.opt.resultdir = '/'.join(params.opt.load_model.split('/')[:-4]) + '/results/'

    params.opt.architecture = storage.Container(storage.Json.load(options.load_model))

    if params.opt.gpu_devices:
        environ["CUDA_VISIBLE_DEVICES"] = params.opt.gpu_devices

def prediction(model, dataset, knn):
    from util.experiments import label2hot
    from util.metrics import h_mean
    from util.storage import Json, Dict_Average_Meter
    import numpy as np

    response = Dict_Average_Meter()

    data_seen = {'x': dataset.test.seen.X,
                 'y': label2hot(dataset.test.seen.Y-1, dataset.n_classes),
                 'a_dict': knn.openset.data,
                 'y_classes': np.zeros(dataset.n_classes),
                 'a': dataset.test.seen.A.continuous,
                 'train_type':model.namespace}
    
    data_unseen = {'x': dataset.test.unseen.X,
                   'y': label2hot(dataset.test.unseen.Y-1, dataset.n_classes),
                   'a_dict': knn.openset.data,
                   'y_classes': np.zeros(dataset.n_classes),
                   'a': dataset.test.unseen.A.continuous,
                   'train_type':model.namespace}

    test_seen, test_unseen = (model.predict(data_seen), model.predict(data_unseen))
    if options.dropout:
        response.set_param("{}/test/seen/vanilla/y_score".format(model.namespace), test_seen)
        response.set_param("{}/test/unseen/vanilla/y_score".format(model.namespace), test_unseen)

        _data = model.predict_mcmc(data_seen)
        response.set_param("{}/test/seen/y_score".format(model.namespace), _data)
        del _data

        _data = model.predict_mcmc(data_unseen)
        response.set_param("{}/test/unseen/y_score".format(model.namespace), _data)
        del _data
    else:
        response.set_param("{}/test/seen/y_score".format(model.namespace), test_seen)
        response.set_param("{}/test/unseen/y_score".format(model.namespace), test_unseen)

    return response.as_dict()


def benchmark(model, dataset, knn):
    from util.experiments import label2hot
    from util.metrics import h_mean
    from util.storage import Json, Dict_Average_Meter
    import numpy as np

    response = Dict_Average_Meter()

    data_seen = {'x': dataset.test.seen.X,
                 'y': label2hot(dataset.test.seen.Y-1, dataset.n_classes),
                 'a_dict': knn.openset.data,
                 'y_classes': np.zeros(dataset.n_classes),
                 'a': dataset.test.seen.A.continuous,
                 'train_type':model.namespace}
    
    data_unseen = {'x': dataset.test.unseen.X,
                   'y': label2hot(dataset.test.unseen.Y-1, dataset.n_classes),
                   'a_dict': knn.openset.data,
                   'y_classes': np.zeros(dataset.n_classes),
                   'a': dataset.test.unseen.A.continuous,
                   'train_type':model.namespace}

    test_seen, test_unseen = (model.evaluate(data_seen), model.evaluate(data_unseen))
    _hmean = h_mean(test_seen['{}_acc'.format(model.get_name())], 
                    test_unseen['{}_acc'.format(model.get_name())])
    response.update_meters("{}/test".format(model.namespace), {'h_mean': _hmean})
    response.update_meters("{}/test/seen".format(model.namespace), test_seen)   
    response.update_meters("{}/test/unseen".format(model.namespace), test_unseen)

    _base = "{}/test/{}/{}_acc".format(model.namespace, '{}', model.namespace)
    print(':: Test-Evaluation: '
        'y(U): {:.3g} | y(S): {:.3g} | H: {:.3g}'.format(response.get_meter(_base.format('unseen')).value(),
                                                         response.get_meter(_base.format('seen')).value(),
                                                         response.get_meter("{}/test/h_mean".format(model.namespace)).value()))
    return response.as_dict()



def main(options, dataset, knn):

    from util.storage import DataH5py, Json
    from util.setup import mkdir
    from routines.aux import get_basic_model

    if options.load_model:
        from .aux import load_model
        model = load_model(options.load_model, options.architecture.namespace)

    try:
        print(":: Testing benchmark")
        results = benchmark(model=model, dataset=dataset, knn=knn)
        DataH5py().save_dict_to_hdf5(dic=results, filename='{}/results.h5'.format(options.output))

        print(":: Saving predictions for future test...")
        predictions = prediction(model=model, dataset=dataset, knn=knn)
        DataH5py().save_dict_to_hdf5(dic=predictions, filename='{}/predictions.h5'.format(options.output))

    except:
        import sys, traceback
        traceback.print_exc(file=sys.stdout)

    return model, None


if __name__ == '__main__':
    print('-'*100)
    print(':: Training file: {}'.format(__file__))
    print('-'*100)
    
    from options.test import TestOptions as Options
    from util import datasets
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    from models import *
    
    try:
        print(':: Seeding to guarantee reproducibility')
        __seed__()
        print(':: Parsing parameters')
        params = Options()
        options = params.parse()
        print('='*50,'\n:: Initializing testing')
        initialize(params)

        print('='*50,'\n:: Loading Dataset')
        dataset, knn = datasets.load(options.datadir)
        print(':: Number of classes: ', dataset.n_classes)

        print('='*50, '\n:: Executing main routines')
        model, results = main(options=options, dataset=dataset, knn=knn)

    except Exception as e:
        import sys, traceback

        traceback.print_exc(file=sys.stdout)