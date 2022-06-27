from nonmlu_ops.base import *
import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd import Function

import sys 
sys.path.append("/R-FCN.pytorch")
from lib.model.roi_crop._ext import roi_crop

class RoICropBackward(Function):
    def forward(self, input1,input2,grad_input1,grad_input2,grad_output):
        roi_crop.BilinearSamplerBHWD_updateGradInput_cuda(input1, input2, grad_input1, grad_input2, grad_output)
        return grad_input1

@registerTensorList("roi_crop_backward")
class RoiCropBackwardTensorList(TensorList):
    pass

@registerOp("roi_crop_backward")
class RoiCropBackwardOp(OpTest):
    def __init__(self, tensor_list, params):
        super().__init__(tensor_list, params)
        self.gradOutput_shape_ = self.tensor_list_.getInputTensor(0).getShape()
        self.grid_shape_ = self.tensor_list_.getInputTensor(1).getShape()
        self.gridInput_tensor = self.tensor_list_.getOutputTensor(0).getShape()

    def compute(self):
        gradOutput_tensor = self.tensor_list_.getInputTensor(0)
        grid_tensor = self.tensor_list_.getInputTensor(1)
        gradInput_tensor = self.tensor_list_.getOutputTensor(0)
        
        data_type = gradOutput_tensor.getDataType().getStr()
        if data_type =="float32":
            go_tensor_NHWC = torch.Tensor(gradOutput_tensor.getData())
            go_tensor_NCHW = go_tensor_NHWC.transpose(2,3).transpose(1,2)
            grad_output = go_tensor_NCHW.float().cuda()

            gi_tensor_NHWC = torch.randn(self.gridInput_tensor)
            gi_tensor_NCHW = gi_tensor_NHWC.transpose(2,3).transpose(1,2)
            input1 = gi_tensor_NCHW.float().cuda()
            grad_input1 = input1.new(input1.size()).zero_()
            
            input2 = torch.Tensor(grid_tensor.getData()).float().cuda()
            grad_input2 = input2.new(input2.size()).zero_()
        else:
            return -1

        RCNN_roi_crop = RoICropBackward()
        result1 = RCNN_roi_crop(Variable(input1), Variable(input2).detach(),Variable(grad_input1), Variable(grad_input2).detach(),Variable(grad_output))
        result_tensor = result1.cpu().data
        result_NHWC = result_tensor.transpose(1,2).transpose(2,3)
        gradInput_tensor.setData(result_NHWC)
        
@registerProtoWriter("roi_crop_backward")
class OpTensorProtoWriter(MluOpProtoWriter):
    pass