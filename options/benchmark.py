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
from .models import ModelsBase
from .dtype import *

class Benchmark(ModelsBase):
    def __init__(self):
        super(Benchmark, self).__init__()

    def initialize(self):
        super(Benchmark, self).initialize()
        self.parser.add_argument('--metric_list', type=str2list, help='Type of attribute',
                                 default=['acc', 
                                          'classifier_loss',
                                          'h_mean'])
        self.parser.add_argument('--exp_directories', type=list, help='Directories on experiments_eccv18 folders "["list of names"]"',
                                 default=['checkpoint', 'results', 'logs'])

        self.parser.add_argument('-arch', '--architecture_file', type=str, default='',
                                 help='Type of attribute to assess')

        self.parser.add_argument('-d', '--domain', type=str, default='openset',
                                 help='[openset, openval, zsl]')

        self.parser.add_argument('--every', type=int, default=5, help='Number of epochs')
        
        self.parser.add_argument('--merge', type=str2bool, default=False, help='Number of epochs')
        self.parser.add_argument('--replace', type=str2bool, default=True, help='Number of epochs')

        self.parser.add_argument('--augm_file', type=str, default=False, help='Filename and path to *.h5')
        self.parser.add_argument('--augm_operation', type=str, default='replace', help="How to augmente the dataset? [replace, merge]")



__OPTION__=Benchmark

if __name__ == '__main__':
    print('-'*100)
    print(':: Testing file: {}'.format(__file__))
    print('-'*100)

    params = GANOptions()
    params.parse()
