from __future__ import print_function
from CNNCalculator import Tensor, CNNCalculator

class MobileNetCalculator(CNNCalculator):
    def __init__(self, width_mul=1):
        super(MobileNetCalculator, self).__init__()
        self.width_mul = width_mul

    '''
    Combination of Conv2d and BatchNorm2d.
    '''
    def ConvBN(self, tensor, out_c, size, stride=1, padding=0, groups=1):
        tensor = self.Conv2d(tensor, out_c, size, stride=stride, padding=padding, groups=groups, bias=False)
        tensor = self.BatchNorm2d(tensor)
        return tensor

    '''
    Depthwise Separable Convolution.
    '''
    def DepthwiseSeparable(self, tensor, out_c, stride):
        in_c = tensor.c
        tensor = self.ConvBN(tensor, in_c, 3, stride=stride, padding=1, groups=in_c)
        tensor = self.ConvBN(tensor, out_c, 1)
        return tensor

    '''
    MobileNet architecture.
    '''
    def MobileNet(self, tensor):
        width_mul = self.width_mul
        tensor = self.ConvBN(tensor, int(32 * width_mul), 3, stride=2, padding=1)
        tensor = self.ReLU(tensor)

        tensor = self.DepthwiseSeparable(tensor, int(64 * width_mul), 1)
        tensor = self.DepthwiseSeparable(tensor, int(128 * width_mul), 2)
        tensor = self.DepthwiseSeparable(tensor, int(128 * width_mul), 1)
        tensor = self.DepthwiseSeparable(tensor, int(256 * width_mul), 2)
        tensor = self.DepthwiseSeparable(tensor, int(256 * width_mul), 1)
        tensor = self.DepthwiseSeparable(tensor, int(512 * width_mul), 2)
        for _ in range(5):
            tensor = self.DepthwiseSeparable(tensor, int(512 * width_mul), 1)
        tensor = self.DepthwiseSeparable(tensor, int(1024 * width_mul), 2)
        tensor = self.DepthwiseSeparable(tensor, int(1024 * width_mul), 1)
        tensor = self.AvgPool2d(tensor, 7)
        tensor = self.Linear(tensor, 1000)
        return tensor

    def calculate(self):
        tensor = Tensor(3, 224, 224)
        tensor = self.MobileNet(tensor)
        print('params: {}, flops: {}'.format(self.params, self.flops))

calculator = MobileNetCalculator(1)
calculator.calculate()
