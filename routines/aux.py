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
_repeat_=100
def __add_parent__():
    import os
    import sys
    sys.path.append(os.path.abspath('../'))
    sys.path.append(os.path.abspath('./'))


def load_model(architecture_file, mtype='base'):
    import models
    from tensorflow import GPUOptions, ConfigProto, Session
    checkdir = '/'.join(architecture_file.split('/')[:-1]) + '/'
    
    print('\n'*2, '-'*_repeat_, '\n:: Open Session\n', '-'*_repeat_, '\n')
    gpu_options = GPUOptions(per_process_gpu_memory_fraction=0.5)
    config=ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = Session(config=config)
    print('\n', '-'*_repeat_)
    
    model = models.__dict__[mtype].__MODEL__()
    pkg= {'model':model, 
          'architecture': architecture_file,
          'dir':checkdir}  
    models.base.__MODEL__.load_architecture(pkg)
    model.set_session(sess)
    model.build(training=False)
    model.load(pkg)
    return model

def __tensorboard_script__(fname='/tmp/tensorboard_script.sh', logidr='/tmp/'):
    with open(fname, 'w') as out:
        out.write('#!/usr/bin/env bash\n')
        out.write('tensorboard --logdir=\'{}\' --port=6006\n'.format(logidr))

def __seed__():
    from tensorflow import set_random_seed
    from numpy.random import seed
    _seed = 53
    seed(_seed)
    set_random_seed(_seed)

def __git_version__():
    import inspect, os
    try:
        with open('./.git/refs/heads/master') as f:
            committag = f.readline()
        file_ = inspect.getfile(inspect.currentframe())  # script filename (usually with path)
        dir_ = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        return {'commit': committag, 'script': file_, 'project': dir_}
    except:
        raise Exception('Not possible to reach project commit versioning!')


def update_metric(model, mflag, _set, answer):
    for key, item in answer.items():
        if key in _set.__dict__:
            _set.__dict__[key].update(item)
            try:
                model.set_summary(tag='{}/{}/metric/{}'.format(model.namespace, mflag, key), 
                                  value=item)
            except:
                pass

def get_basic_model(value):
    import models
    return {'classifier': models.classifier,
            'regressor': models.regressor,
            'generator': models.generators,
            'discriminator': models.discriminators}[value]