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
from .dtype import *

import argparse

class Base():
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        from time import strftime
        # Code setupt directories
        self.parser.add_argument('-r', '--root', type=str, help='Setup experiment folder (full)', default='/tmp')
        self.parser.add_argument('-n', '--namespace', type=str, help='Experiment folder', default='')
        self.parser.add_argument('-broot','--baseroot', type=str, help='Parent directory for experimenta folder', default='/tmp')
        self.parser.add_argument('-aroot', '--auxroot', type=str, help='Extra directory or file', default='/tmp/')
        
        # Computer setup
        self.parser.add_argument('--gpu_devices', type=str, help='Number of GPU cores required', default=False)
        self.parser.add_argument('--gpu_memory', type=float, help='Memory to required for each core', default=0.9)

        # Utils
        self.parser.add_argument('--timestamp', type=str, help='Timestamp',
                                 default='{}'.format(strftime("%d%m%y_%H%M%S")))

        self.parser.add_argument('-db', '--dbname', type=str, help='Dataset root', default='AWA1')
        self.parser.add_argument('-droot', '--dataroot', type=str, help='Dataset root', default='/var/scientific/data/eccv18/')
        self.parser.add_argument('-ddir', '--datadir', type=str, help='Dataset root path for file', default='')
        
        self.initialized = True


    def parse(self, verbose=False):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        if self.opt.datadir == '':
            self.opt.datadir = '{}/{}/'.format(self.opt.dataroot, self.opt.dbname).lower()
        
        args = vars(self.opt)

        if verbose:
            self.print()

        return self.opt

    def save(self, root):
        try:
            from util.storage import Json
            Json().save(self.as_dict(), root)
        except:
            raise Exception('Error: Json:save: not possible to save')

    def print(self):
        args = vars(self.opt)
        print('\n\n','------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------','\n\n')

    def as_dict(self):
        if self.initialized:
            return self.opt.__dict__

    def keys(self):
        return list(self.__dict__.keys())


_OPTION_=Base

if __name__ == '__main__':
    print('-'*100)
    print(':: Testing file: {}'.format(__file__))
    print('-'*100)

