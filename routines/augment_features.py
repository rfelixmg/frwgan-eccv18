_repeat_=100

from .aux import load_model, __add_parent__

def get_new_features(model, domain, num_features):
    from numpy import array
    answer = {'X': array([]),
              'Y': array([]),
              'A': {'continuous' : array([])}}

    for (input_a, input_y) in domain:
        batch_a = array([input_a] * num_features)
        batch_y = array([input_y] * num_features)
        batch_z = model.get_noise(shape=[num_features, input_a.shape[0]])
        data = {'a': batch_a,
                'z': batch_z}
        _out = model(data)
        answer['X'] = model.merge_array(answer['X'], _out)
        answer['Y'] = model.merge_array(answer['Y'], batch_y)
        answer['A']['continuous'] = model.merge_array(answer['A']['continuous'], batch_a)
    out = answer

    return out

def save_data(_dataset):
    new_dataset = _dataset['train']
    from numpy import savetxt
    from util.storage import DataH5py
    from util.tensors import merge_array

    try:
        if not options.merge:
            DataH5py().save_dict_to_hdf5(_dataset, '{}/{}'.format(options.outdir, options.outfile))
        else:
            DataH5py().save_dict_to_hdf5(new_dataset, '{}/_generated_{}'.format(options.outdir, options.outfile))
            dataset.train.X = merge_array(dataset.train.X, 
                                          new_dataset['X']) 
            
            dataset.train.Y = merge_array(dataset.train.Y, 
                                          new_dataset['Y']) 
            
            dataset.train.A.continuous = merge_array(dataset.train.A.continuous, 
                                                     new_dataset['A']['continuous'])
            dataset.info = _dataset['info']
            DataH5py().save_dict_to_hdf5(dataset, '{}/{}'.format(options.outdir, options.outfile))

        if options.save_numpy:
            savetxt('{}/X.npy'.format(options.outdir), new_dataset['X'])
            savetxt('{}/Y.npy'.format(options.outdir), new_dataset['Y'])
            savetxt('{}/A.npy'.format(options.outdir), new_dataset['A']['continuous'])

        try:
            from os import symlink
            symlink('{}/knn.h5'.format(options.datadir),'{}/knn.h5'.format(options.outdir))
        except Exception as e:
            from warnings import warn
            warn("\n:: [warning] Link already exist")
            print(e)

    except Exception as e:
        import sys, traceback
        traceback.print_exc(file=sys.stdout)

def get_architecture(architecture_file):

    if architecture_file[-4:] == 'json':
        return options.architecture_file
    else:
        from util.files import list_directories
        _epochs = list_directories(architecture_file + '/', False)
        _epochs.remove('last_epoch')
        if len(_epochs) > 1:
            from numpy import sort

            available_epochs = list(sort([int(_value.split('_')[1]) for _value in _epochs]))

            _choice = -99
            while(_choice not in available_epochs):
                print('-'*_repeat_, '\n:: Please select epoch to generate features:')
                print('Epochs available: ', available_epochs)
                _choice = int(input('Choice: '))
            _epoch = _epochs[0].split('_')
            _epoch[1] = str(_choice)
            _epoch = '_'.join(_epoch)
        else:
            _epoch = _epochs[0]
        return '{}/{}/architecture.json'.format(architecture_file, _epoch)


def main(dataset, knn):

    import numpy as np
    from util.tensors import merge_dict

    print('-'*_repeat_, "\n:: Loading model\n")
    architecture_file = get_architecture(options.architecture_file[0])
    model = load_model(architecture_file)
    
    new_dataset = {'train':{'X': np.array([]),
                            'Y': np.array([]),
                            'A': {'continuous': np.array([])}},
                   'info': {'dataset': options.dbname,
                            'num_features' : str(options.num_features),
                            'domain' : str(options.domain),
                            'architecture' : str(options.architecture_file),
                            'architecture_file' : str(architecture_file),
                            'timestamp': options.timestamp}}

    
    for _domain, _num in zip(options.domain, options.num_features):
        domain = {'unseen':zip(knn.zsl.data, knn.zsl.ids),
                  'seen':zip(knn.openval.data, knn.openval.ids),
                  'openset':zip(knn.openset.data, knn.openset.ids)}[_domain]

        print('-'*_repeat_, "\n:: Generating features [{}:{}]".format(_domain, _num))
        _db = get_new_features(model, domain, _num)
        new_dataset['train'] = merge_dict(new_dataset['train'], _db)

    print('-'*_repeat_, "\n:: Saving generated dataset")
    save_data(new_dataset)

    return new_dataset

if __name__ == '__main__':
    try:
        __add_parent__()
        from options.generator import __OPTION__ as Option
        from util import datasets
        from models import *


        print('-'*_repeat_)
        print(':: Initializing: {}'.format(__file__))
        print('-'*_repeat_)

        params = Option()
        options = params.parse()

        print('-'*_repeat_,'\n:: Loading Dataset')
        dataset, knn = datasets.load(options.datadir)
        
        print('-'*_repeat_,'\n:: Running main')
        db_answer = main(dataset, knn)
        print('-'*_repeat_,'\n\n:: Finish...')
        print(':: End of session\n\n')

    except Exception as e:
        import sys, traceback
        traceback.print_exc(file=sys.stdout)