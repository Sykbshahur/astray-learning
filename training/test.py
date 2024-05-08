from detectors import DETECTOR
import torch
from thop import profile
model_class = DETECTOR['ucf_oneStage_res34']
model = model_class()
flops, params = profile(model)
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))