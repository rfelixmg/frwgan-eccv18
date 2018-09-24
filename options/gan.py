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

class GANOptions(Base):
    def __init__(self):
        super(GANOptions, self).__init__()

    def initialize(self):
        super(GANOptions, self).initialize()
        self.parser.add_argument('--metric_list', type=str2list, help='Type of attribute',
                                 default=['acc', 
                                          'classifier_loss', 
                                          'discriminator_loss', 
                                          'generator_loss',
                                          'discriminator_real',
                                          'discriminator_fake',
                                          'discriminator_grad',
                                          'regressor_loss'])

        self.parser.add_argument('--checkpoint_metric', type=str, default='c_loss_val',
                                 help='Type of attribute to assess')

        self.parser.add_argument('--architecture_file', type=str, default='./src/architecture/gan_awa.json',
                                 help='Type of attribute to assess')


        self.parser.add_argument('--train_cls', type=str2bool, help='Pre-train classifier?',
                                 default=False)

        self.parser.add_argument('--train_reg', type=str2bool, help='Pre-train regressor?',
                                 default=False)

        self.parser.add_argument('--train_gan', type=str2bool, help='Train GAN',
                                 default=False)

        # Dataset
        self.parser.add_argument('--att_type', type=str, default='continuous', help='Type of attribute')



if __name__ == '__main__':
    print('-'*100)
    print(':: Testing file: {}'.format(__file__))
    print('-'*100)

    params = ExperimentOptions()
    params.parse()
