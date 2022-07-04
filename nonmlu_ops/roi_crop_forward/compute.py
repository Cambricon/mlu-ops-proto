from nonmlu_ops.base import *
import numpy as np
import torch
from torch.autograd import Variable

import sys 
sys.path.append("/R-FCN.pytorch")
from lib.model.roi_crop.functions.roi_crop import RoICropFunction

@registerTensorList("roi_crop_forward")
class RoiCropForwardTensorList(TensorList):
    pass

@registerOp("roi_crop_forward")
class RoiCropForwardOp(OpTest):
    def __init__(self, tensor_list, params):
        super().__init__(tensor_list, params)
        self.input_shape_ = self.tensor_list_.getInputTensor(0).getShape()
        self.grid_shape_ = self.tensor_list_.getInputTensor(1).getShape()
        self.out_tensor = self.tensor_list_.getOutputTensor(0).getShape()

    def compute(self):
        input_tensor = self.tensor_list_.getInputTensor(0)
        grid_tensor = self.tensor_list_.getInputTensor(1) 
        
        out_tensor = self.tensor_list_.getOutputTensor(0)
        data_type = input_tensor.getDataType().getStr()
        if data_type =="float32":
            x_tensor_NHWC = torch.Tensor(input_tensor.getData())
            x_tensor_NCHW = x_tensor_NHWC.transpose(2,3).transpose(1,2)
            x_tensor = x_tensor_NCHW.float().cuda()
            grid = torch.Tensor(grid_tensor.getData()).float().cuda()
        else:
            return -1
        
        rcnn_roi_crop = RoICropFunction()
        result = rcnn_roi_crop(Variable(x_tensor), Variable(grid).detach())
        result_tensor = result.cpu().data
        result_NHWC = result_tensor.transpose(1,2).transpose(2,3)
        out_tensor.setData(result_NHWC)
        
@registerProtoWriter("roi_crop_forward")
class OpTensorProtoWriter(MluOpProtoWriter):
    pass