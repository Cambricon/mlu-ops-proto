import torch
import numpy as np
from nonmlu_ops.base import *

import sys
sys.path.append("/R-FCN.pytorch")
from lib.model.psroi_pooling._ext import psroi_pooling

@registerTensorList("psroipool_backward")
class PsRoiPoolTensorList(TensorList):
    def generateData(self):
        '''
        For generating data inherite this class and rewrite function generateData.
        If tensor has attribute filename, data will be loaded from file,
        else will generate random data.
        '''
        for idx in range(3):
            input_tensor = (self.input_tensors_)[idx]
            if input_tensor.filename_:
                shape = input_tensor.shape_
                dtype = input_tensor.getDataType()
                assert not dtype.isComplex(), "complex type do not support generate data from file"
                dtype_str = dtype.getNumpyStr()
                file_data = np.genfromtxt(input_tensor.filename_, dtype=dtype_str).reshape(shape)
                input_tensor.getDataNode().setData(file_data)
            else:
                if (idx == 0):
                    RandomData(input_tensor).random()
                elif (idx == 1):
                    mappingChannel_array = []
                    mappingChannels_shape = input_tensor.shape_
                    roi_num = mappingChannels_shape[0]
                    pooled_height = mappingChannels_shape[1]
                    pooled_width = mappingChannels_shape[2]
                    output_dim = mappingChannels_shape[3]
                    for b in range(0, roi_num):
                        for i in range(0, pooled_height * pooled_width):
                            for j in range(0, output_dim):
                                value = i + j * pooled_height * pooled_width
                                mappingChannel_array.append(value)
                    mappingChannels_narray = (np.array(mappingChannel_array)).reshape \
                                                (roi_num, pooled_height, pooled_width, output_dim)
                    input_tensor.getDataNode().setData(mappingChannels_narray)
                else:
                    shape_roi = input_tensor.shape_
                    bottom_grad_shape = ((self.output_tensors_)[0]).shape_
                    roi = []
                    # generate roi data first,the rios_type is ROI_BATCHID_CONRER
                    roi_num = int(shape_roi[0])
                    roi_offset = int(5)
                    batch = bottom_grad_shape[0]
                    input_h = bottom_grad_shape[1]
                    input_w = bottom_grad_shape[2]
                    for roi_id in range(0, int(roi_num)):
                        roi_tmp = []
                        roi_x_start = random.random() * float(input_w) * float(input_w)
                        roi_y_start = random.random() * float(input_h) * float(input_h)
                        roi_x_end = random.random() * (float(input_w) * float(input_w) - roi_x_start) + roi_x_start
                        roi_y_end = random.random() * (float(input_h) * float(input_h) - roi_y_start) + roi_y_start
                        if roi_id % 4 == 0:
                            roi_x_start = -random.random() * float(input_w) * float(input_w)
                        roi_tmp.append((int)(random.randint(0, batch - 1)))
                        roi_tmp.append(format(roi_x_start, "0.5f"))
                        roi_tmp.append(format(roi_y_start, "0.5f"))
                        roi_tmp.append(format(roi_x_end, "0.5f"))
                        roi_tmp.append(format(roi_y_end, "0.5f"))
                        roi.extend(roi_tmp)

                    roi_array = (np.array(roi)).reshape(roi_num, roi_offset)
                    input_tensor.getDataNode().setData(roi_array)

@registerOp("psroipool_backward")
class PsRoiPooBackwardlOp(OpTest):
    def __init__(self, tensorlist, params):
        super().__init__(tensorlist, params)
        self.topGrad = self.tensor_list_.getInputTensor(0)
        self.mappingChannel = self.tensor_list_.getInputTensor(1)
        self.rois = self.tensor_list_.getInputTensor(2)
        self.bottomGrad = self.tensor_list_.getOutputTensor(0)
        self.spatial_scale = self.params_.get("spatial_scale")
        self.output_dim = self.params_.get("output_dim")
        self.pooled_height = self.params_.get("pooled_height")
        self.pooled_width = self.params_.get("pooled_width")

    def compute(self):
        # param check
        bottomGrad_shape = self.bottomGrad.shape_
        batch = bottomGrad_shape[0]
        height = bottomGrad_shape[1]
        width = bottomGrad_shape[2]
        channel = bottomGrad_shape[3]
        roi_shape = self.rois.shape_
        rois_num = roi_shape[0]
        rois_offset = roi_shape[1]
        spatial_scale = self.spatial_scale
        output_dim = self.output_dim
        pooled_height = self.pooled_height
        pooled_width = self.pooled_width
        assert (channel == pooled_height * pooled_width * output_dim)
        assert (rois_offset == 5)
        assert (pooled_width == pooled_height)
        assert (len(self.tensor_list_.getInputTensors()) == 3)
        assert (len(self.tensor_list_.getOutputTensors()) == 1)
        
        topGrad_layout = self.topGrad.layout_
        bottomGrad_layout = self.bottomGrad.layout_
        mappingChannel_layout = self.mappingChannel.layout_
        
        assert topGrad_layout == Layout.NHWC , "ERROR: the input data layout only support NHWC"
        assert mappingChannel_layout == Layout.NHWC , "ERROR: the input data layout only support NHWC"
        assert bottomGrad_layout == Layout.NHWC , "ERROR: the output data layout only support NHWC"
        
        topGrad_tensor = torch.from_numpy(self.topGrad.getData())
        mappingChannel_tensor = torch.from_numpy(self.mappingChannel.getData())
        torch_roi_tensor = torch.from_numpy(self.rois.getData()).cuda()

        # the input data must be transposed to NCHW
        topGrad_tensor_trans = topGrad_tensor.permute(0, 3, 1, 2).cuda()
        mappingChannel_tensor_trans = mappingChannel_tensor.permute(0, 3, 1, 2).cuda()
        bottomGrad_data = torch.zeros(batch, channel, height, width).cuda()
        psroi_pooling.psroi_pooling_backward_cuda(pooled_height, pooled_width, spatial_scale, output_dim, topGrad_tensor_trans, torch_roi_tensor, bottomGrad_data, mappingChannel_tensor_trans)
        bottomGrad_result = bottomGrad_data.permute(0, 2, 3, 1)
        self.bottomGrad.setData(bottomGrad_result.cpu())
        
@registerProtoWriter("psroipool_backward")
class PsRoiPoolBackwardProtoWriter(MluOpProtoWriter):
    def dumpOpParam2Node(self):
        param_node = self.proto_node_.psroipool_backward_param
        param_node.spatial_scale = self.op_params_.get("spatial_scale")
        param_node.output_dim = self.op_params_.get("output_dim")
        param_node.pooled_height = self.op_params_.get("pooled_height")
        param_node.pooled_width = self.op_params_.get("pooled_width")
        