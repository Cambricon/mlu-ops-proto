import numpy as np
import torch
from nonmlu_ops.base import *
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext('_ext', ['three_interpolate_forward'])


@registerTensorList("three_interpolate_forward")
class ThreeInterpolateForwardTensorList(TensorList):
    pass

@registerOp("three_interpolate_forward")
class ThreeInterpolateForwardOp(OpTest):
    def __init__(self, tensor_list, params):
        super().__init__(tensor_list, params)
        self.features = self.tensor_list_.getInputTensor(0)
        self.idx = self.tensor_list_.getInputTensor(1)
        self.weight = self.tensor_list_.getInputTensor(2)
        self.out_tensor = self.tensor_list_.getOutputTensor(0)

    def compute(self):
        features_shape = self.features.shape_
        idx_shape = self.idx.shape_
        weight_shape = self.weight.shape_
        b = features_shape[0]
        c = features_shape[1]
        m = features_shape[2]
        n = idx_shape[1]
        assert (features_shape[0] == idx_shape[0]) and (features_shape[0] == weight_shape[0]), "input batch should be same."
        assert (idx_shape[1] == weight_shape[1]), "idx and weight's n should be same."
        assert (idx_shape[2] == weight_shape[2]) and (idx_shape[2] == 3), "idx and weight's third dim should be same as 3."
        features_dtype = self.features.getDataType()
        idx_dtype = self.idx.getDataType()
        weight_dtype = self.weight.getDataType()
        output_dtype = self.out_tensor.getDataType()
        print(features_dtype.getNumpyStr())
        assert (features_dtype == weight_dtype) and (features_dtype == output_dtype), "features, weight and output's data type should be same."
        assert (features_dtype == DataType.FLOAT32 or features_dtype == DataType.FLOAT16, "features, weight and output's data type should be float32 or float16.")
        assert (idx_dtype == DataType.INT32, "idx's data type should only be int32.")

        features = torch.from_numpy(self.features.getData()).cuda()
        indices  = torch.from_numpy(self.idx.getData()).cuda()
        weight   = torch.from_numpy(self.weight.getData()).cuda()

        result = features.new_zeros(b,c,n).cuda()
        ext_module.three_interpolate_forward(
            features, indices, weight, result, b=b, c=c, m=m, n=n)
        result_tensor = result.cpu().data
        self.out_tensor.setData(result_tensor)



@registerProtoWriter("three_interpolate_forward")
class OpTensorProtoWriter(MluOpProtoWriter):
    pass
