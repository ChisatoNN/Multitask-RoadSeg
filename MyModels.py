from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv, C3, TransformerLayer, TransformerBlock, C3TR, ASPP, C2f, C3GC, SPPF, ResNeXtBottleneckCSP, ResNeXtBottleneck, SELayer
from torch.nn import Upsample

'''
default 
 lr=0.001 
 lf=0.2 
 epoch=30 
LOSS:
  BOX_GAIN: 0.05
  CLS_GAIN: 0.35
  CLS_POS_WEIGHT: 1.0
  DA_SEG_GAIN: 0.5
  FL_GAMMA: 0.0
  LL_IOU_GAIN: 0.4
  LL_SEG_GAIN: 0.4
'''
YOLOPdeep = [
[ 24, 36, 48],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16         #Encoder

[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detection head 24

[ 16, Conv, [256, 128, 3, 1]],   #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ [-1, 2], Concat, [1]],    #27
[ -1, BottleneckCSP, [192, 256, 1, False]],   #28
[ -1, Conv, [256, 128, 3, 1]],    #29
[ -1, BottleneckCSP, [128, 64, 1, False]],  #30
[ -1, Conv, [64, 32, 3, 1]],    #31
[ -1, Upsample, [None, 2, 'nearest']],  #32
[ -1, Conv, [32, 16, 3, 1]],  #33
[ -1, BottleneckCSP, [16, 8, 1, False]], #34
[ -1, Upsample, [None, 2, 'nearest']],  #35
[ -1, Conv, [8, 2, 3, 1]], #36 Driving area segmentation head

[ 16, Conv, [256, 128, 3, 1]],   #37
[ -1, Upsample, [None, 2, 'nearest']],  #38
[ [-1, 2], Concat, [1]],    #39
[ -1, BottleneckCSP, [192, 256, 1, False]],   #40
[ -1, Conv, [256, 128, 3, 1]],    #41
[ -1, BottleneckCSP, [128, 64, 1, False]],  #42
[ -1, Conv, [64, 32, 3, 1]],    #43
[ -1, Upsample, [None, 2, 'nearest']],  #44
[ -1, Conv, [32, 16, 3, 1]],  #45
[ -1, BottleneckCSP, [16, 8, 1, False]], #46
[ -1, Upsample, [None, 2, 'nearest']],  #47
[ -1, Conv, [8, 2, 3, 1]],  #48 Lane line segmentation head
]