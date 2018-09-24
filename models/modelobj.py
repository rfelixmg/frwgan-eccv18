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

class Print(object):
    def __init__(self):
        pass

    @classmethod
    def print_inline(self, x):
        from sys import stdout
        stdout.write('\r{}               '.format(x))
        stdout.flush()


class ModelObject(object):
    def __init__(self, hparams):
        self.counter = 0
        self.__set_dict__(hparams)
        self.printer = Print.print_inline

    def reset_counter(self):
        self.counter = 0

    def __set_dict__(self, data):
        for key, value in data.items():
            self.__dict__[key] = value

    def get_subparam(self, tree, data):
        levels = data.split('/')
        if(len(levels) > 1):
            return self.get_subparam(tree[levels[0]], '/'.join(levels[1:]))
        else:
            return tree[data]

    def get_param(self, data):
        return self.get_subparam(self.__dict__, data)
    
    def set_writer(self, root):
        import tensorflow as tf
        self.writer = tf.summary.FileWriter(root, self.sess.graph)
        self.writer.flush()
    
    def set_summary(self, tag, value, stype='summary'):
        if stype == 'summary':
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.writer.add_summary(summary, self.counter)
        elif stype == 'histogram':
            self.__summary_histogram__(tag, value)
        elif stype == 'text':
            self.__summary_text__(tag, value)
    
    def __summary__(self, summary, merges=False):
        summaries = [tf.summary.scalar(name=key, tensor=value) for key, value in summary.items()]
        if merges is not False:
            summaries.append(merges)
        return tf.summary.merge(summaries)

    def __summary_text__(self, tag, value):
        text_tensor = tf.make_tensor_proto(str(value), dtype=tf.string)
        meta = tf.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"
        summary = tf.Summary()
        summary.value.add(tag=tag, metadata=meta, tensor=text_tensor)
        self.writer.add_summary(summary, self.counter)

    def __summary_histogram__(self, tag, values, bins=100):
        import numpy as np

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, self.counter)
        self.writer.flush()

    def __summary_features__(self, tag, values, bins=100):
        import numpy as np

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, self.counter)
        self.writer.flush()

    @classmethod
    def list_ids(self, x, shuffle=True):
        """
        list ids: get a matrix and return a shuffle list of positions
        :param x: 
        :param shuffle: 
        :return: 
        """
        from numpy import array
        from numpy.random import permutation
        dim_x = x.shape[0]
        # ids_ = np.array(range(dim_x))

        if shuffle:
            ids_ = permutation(dim_x)
        else:
            ids_ = array(range(dim_x))

        return ids_, dim_x

    @classmethod
    def merge_array(self, x, y, axis=0):
        from numpy import size,atleast_1d, concatenate
        if not size(x):
            return atleast_1d(y)
        elif size(x) and size(y):
            return concatenate([x, atleast_1d(y)], axis)
        elif size(y):
            return atleast_1d(y)
        else:
            return atleast_1d([])

    def next_batch(self, data, batch_size, shuffle=True):
        import numpy as np
        
        ids_, dim_x = self.list_ids(x=data[self.placeholder_input[0]], shuffle=shuffle)
        for batch in range(0, dim_x, batch_size):
            ii_ = ids_[batch: batch_size + batch]
            pck = (ii_, batch, dim_x)
            yield pck

    def get_noise(self, shape, mean=0.0, var=1.0):
        import numpy as np
        return np.random.normal(loc=mean, scale=var, size=shape)

    @classmethod
    def accuracy_per_class(self, predict_label, true_label, classes):
        '''    
        :param predict_label: output of model (matrix)
        :param true_label: labels from dataset (array of integers)
        :param classes: class labels list() 
        :return: 
        '''
        from numpy import sum, float, array
        if isinstance(classes, int):
            nclass = classes
            classes = range(nclass)
        else:
            nclass = len(classes)
        
        acc_per_class = []
        for i in range(nclass):
            idx = true_label == classes[i]
            if idx.sum() != 0:
                acc_per_class.append(sum(true_label[idx] == predict_label[idx]) / float(idx.sum()))
        if len(acc_per_class) == 0:
            return 0.
        
        return array(acc_per_class).mean()