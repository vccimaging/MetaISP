from .base_options import BaseOptions
import argparse
from options.base_options import str2bool


class SingleOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        #
        parser.add_argument('--path_model', type=str, default="", help='Path to pth file.')
        parser.add_argument('--path_image', type=str, default="", help='Path to image. dng and jpg should share same name')
        parser.add_argument('--device', type=str, default='xiaomi', help='select between xiaomi or iphone')
        parser.add_argument('--img_size', type=int, default='2688', help='full image size')
        parser.add_argument('--bilinear_size', type=int, default='448', help='bilinear downsampled image size')
        parser.add_argument('--pre_trained_devices', type=int, default='3', help='number of devices that the model was trained on')
        parser.single_image = True
        self.isTrain = False
        return parser
