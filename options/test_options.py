from .base_options import BaseOptions
from options.base_options import str2bool

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--sname',type=str, default='18')
        self.isTrain = False
        return parser
