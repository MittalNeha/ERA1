import torch.nn as nn
import torch.nn.functional as F

#depthwise separable convolution
class DepthwiseSeparable(nn.Module):
  def __init__(self, in_ch, out_ch, stride=1):
    super(DepthwiseSeparable, self).__init__()
    self.in_chan = in_ch
    self.out_chan = out_ch

    self.depthwise = nn.Sequential(
          nn.Conv2d(in_channels=self.in_chan, out_channels=self.in_chan, kernel_size=(3, 3), padding=1, stride=stride, groups=self.in_chan, bias=False),
          #pointwise
          nn.Conv2d(in_channels=self.in_chan, out_channels=self.out_chan, kernel_size=(1,1)))

  def forward(self, x):
    x = self.depthwise(x)
    return x

class Net(nn.Module):
    def __init__(self, drop=0.025):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),  # Input: 32x32x3 | Output: 32x32x32 | RF: 3x3
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(drop),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),   # Input: 32x32x32 | Output: 32x32x32 | RF: 5x5
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(drop),
        )

        self.transblock1 =  nn.Sequential(
            #Stride 2 conv
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=2, bias=False),  # Input: 32x32x32 | Output: 16x16x32 | RF: 7x7
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(drop),

        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),   # Input: 16x16x32 | Output: 16x16x64 | RF: 11x11
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(drop),  

        ) 

        self.transblock2 =  nn.Sequential(
            #Stride 2 conv
            DepthwiseSeparable(64,64,2), # Input: 16x16x64 | Output: 8x8x64 | RF: 15x15
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(drop),
        )

        self.convblock3 = nn.Sequential(
            DepthwiseSeparable(64, 128), # Input: 8x8x64 | Output: 8x8x128 | RF: 23x23
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(drop),  
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), bias=False), # Input: 8x8x128 | Output: 8x8x32 | RF: 23x23
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(drop), 

        ) 

        self.transblock3 =  nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, dilation=2, bias=False), # Input: 8x8x128 | Output: 6x6x32 | RF: 39x39
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, dilation=2, bias=False), # Input: 6x6x32 | Output: 4x4x32 | RF: 55x55
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(drop),
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), # Input: 4x4x32 | Output: 4x4x32 | RF: 63x63
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(3, 3), padding=1, bias=False), # Input: 4x4x32 | Output: 4x4x10 | RF: 71x71

        ) 

        self.gap = nn.AvgPool2d(4)

    def forward(self, x):
        x =  self.transblock1(self.convblock1(x))
        x =  self.transblock2(self.convblock2(x))
        x =  self.transblock3(self.convblock3(x))
        x =  self.convblock4(x)
        x =  self.gap(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=-1)
