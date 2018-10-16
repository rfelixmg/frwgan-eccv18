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
from .base import Base
from .dtype import *

class GeneratorOptions(Base):
    def __init__(self):
        super(GeneratorOptions, self).__init__()

    def initialize(self):
        super(GeneratorOptions, self).initialize()
        self.parser.add_argument('--architecture_file', type=str2strlist, default='',
                                 help='Architecture json file')

        self.parser.add_argument('--num_features', type=str2intlist, default=300, help='Num of features to generate')
        self.parser.add_argument('--iter', type=str2intlist, default=10, help='Num of features to generate')
        self.parser.add_argument('--domain', type=str2strlist, default='unseen', help='Generate from given domain: (seen, unseen, openset)')
        self.parser.add_argument('--outdir', type=str, default='/tmp/', help='Default directory to save features')
        self.parser.add_argument('--outfile', type=str, default='data.h5', help='Filename to save generated features')
        
        self.parser.add_argument('--verbose', type=str2bool, default=False, help='Print progress?')
        self.parser.add_argument('--specs', type=str, default=False, help='Specification json file')
        
        self.parser.add_argument('--merge', type=str2bool, default=False, help='Option to merge fake to original dataset')
        self.parser.add_argument('--save_numpy', type=str2bool, default=False, help='Option to save numpy matrix obj')


__OPTION__=GeneratorOptions

if __name__ == '__main__':
    print('-'*100)
    print(':: Testing file: {}'.format(__file__))
    print('-'*100)

    params = GeneratorOptions()
    params.parse()
