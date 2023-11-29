from .base_options import BaseOptions


class MetricOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--path_pred', type=str, default="", help='Path to pth file.')
        parser.add_argument('--path_gt', type=str, default="", help='Dataset')
        parser.add_argument('--meta', type=str, default="", help='select between xiaomi or iphone')
        parser.add_argument('--save_name', type=str, default='save.csv', help='select between xiaomi or iphone')
        self.isTrain = False
        return parser
