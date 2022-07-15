import os
import torch
import numpy as np
from nonmlu_ops.base import *

import sys
sys.path.append("/R-FCN.pytorch")
from lib.model.psroi_pooling._ext import psroi_pooling

@registerTensorList("psroipool_forward")
class PsRoipoolTensorList(TensorList):
    def generateData(self):
        '''
        For generating data inherite this class and rewrite function generateData.

        If tensor has attribute filename, data will be loaded from file,
        else will generate random data.
        '''
        for input_tensor in self.input_tensors_:
            if input_tensor.filename_:
                shape = input_tensor.shape_
                dtype = input_tensor.getDataType()
                assert not dtype.isComplex(), "complex type do not support generate data from file"
                dtype_str = dtype.getNumpyStr()
                file_data = np.genfromtxt(input_tensor.filename_, dtype=dtype_str).reshape(shape)
                input_tensor.getDataNode().setData(file_data)
            else:
                if (input_tensor == (self.input_tensors_)[0]):
                    RandomData(input_tensor).random()
                else:
                    shape_roi = input_tensor.shape_;
                    input_shape = ((self.input_tensors_)[0]).shape_;
                    roi = []
                    # the rios_type is ROI_BATCHID_CONRER
                    roi_num = int((shape_roi[0] * shape_roi[1]) / 5)
                    roi_offset = int(5)
                    batch = input_shape[0]
                    roi_num_each_batch = roi_num / batch
                    input_h = input_shape[1]
                    input_w = input_shape[2]
                    for roi_id in range(0, int(roi_num)):
                        roi_tmp = []
                        roi_x_start = random.random() * float(input_w) * float(input_w)
                        roi_y_start = random.random() * float(input_h) * float(input_h)
                        roi_x_end = random.random() * (float(input_w) * float(input_w) - roi_x_start) + roi_x_start
                        roi_y_end = random.random() * (float(input_h) * float(input_h) - roi_y_start) + roi_y_start
                        roi_tmp.append((int)(random.randint(0, batch - 1)))
                        roi_tmp.append(format(roi_x_start, "0.5f"))
                        roi_tmp.append(format(roi_y_start, "0.5f"))
                        roi_tmp.append(format(roi_x_end, "0.5f"))
                        roi_tmp.append(format(roi_y_end, "0.5f"))
                        roi.extend(roi_tmp)

                    roi_array = (np.array(roi)).reshape(roi_num, roi_offset)
                    input_tensor.getDataNode().setData(roi_array)

@registerOp("psroipool_forward")
class PsRoipoolOp(OpTest):
    def __init__(self, tensorlist, params):
        super().__init__(tensorlist, params)
        self.input = self.tensor_list_.getInputTensor(0).getData()
        self.roi = self.tensor_list_.getInputTensor(1).getData()
        self.input_shape =  self.tensor_list_.getInputTensor(0).shape_
        self.roi_shape = self.tensor_list_.getInputTensor(1).shape_
        self.spatial_scale = self.params_.get("spatial_scale")
        self.group_size = self.params_.get("group_size")
        self.output_dim = self.params_.get("output_dim")
        self.pooled_height = self.params_.get("pooled_height")
        self.pooled_width = self.params_.get("pooled_width")

    def compute(self):
        # param check
        input_shape = self.input_shape
        roi_shape = self.roi_shape
        batch = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channel = input_shape[3]
        rois_offset = roi_shape[1]
        rois_num = roi_shape[0]
        spatial_scale = self.spatial_scale
        group_size = self.group_size
        output_dim = self.output_dim
        pooled_height = self.pooled_height
        pooled_width = self.pooled_width
        if (channel != group_size * group_size * output_dim):
            print("Error: the channel must be equal to group_size * group_size * output_dim, channel is = %d, group_size is %d, output_dim is %d" %(channel, group_size, output_dim))
        if rois_offset != 5:
            print("Error: the roi input second dim must be equal to 5, but now is %d" %rois_offset)
        if (group_size != pooled_height):
            print("Error: the group_size must be equal to pooled_height, but now group_size is %d, pooled_height is %d" %(group_size, pooled_height))
        if (pooled_width != pooled_height):
            print("Error: the pooled_width must be equal to pooled_height, but now pooled_width is %d, pooled_height is %d" %(pooled_width, pooled_height))
        if (len(self.tensor_list_.getInputTensors()) != 2):
            print("Error: the input tensors must be equal to 2, but now is %d" %(len(self.tensor_list_.getInputTensors())))
        if (len(self.tensor_list_.getOutputTensors()) != 2):
            print("Error: the output tensors must be equal to 2, but now is %d" %(len(self.tensor_list_.getOutputTensors())))

        inputs : list[Tensor] =  self.tensor_list_.getInputTensors()
        input_tensor = inputs[0]
        input_rois_tensor = inputs[1]
        output_node = self.tensor_list_.getOutputTensor(0)
        mapping_channes = self.tensor_list_.getOutputTensor(1)
        rois_layout = input_rois_tensor.layout_
        input_layout = input_tensor.layout_
        output_layout = output_node.layout_
        mapping_channes_layout = mapping_channes.layout_
        if (input_layout != Layout.NHWC):
            print("ERROR: the input data layout only support NHWC, but now is " + input_layout)
        if (output_layout != Layout.NHWC):
            print("ERROR: the outpu data layout only support NHWC, but now is " + output_layout)
        if (mapping_channes_layout != Layout.NHWC):
            print("ERROR: the outpu data layout only support NHWC, but now is " + mapping_channes_layout)
        input_data = inputs[0].getData()
        # the input data is NHWC, but the torch input must be NCHW
        input_data_tensor = torch.from_numpy(input_data)
        # the roi data must be [K, 5]
        roi_data = inputs[1].getData()
        torch_roi_tensor = torch.from_numpy(roi_data).cuda()
        # the input data must be trans to NCHW
        input_data_tensor_trans = input_data_tensor.permute(0, 3, 1, 2).cuda()
        top_data = torch.zeros(rois_num, output_dim, pooled_height, pooled_width)
        mapping_channes_data = torch.IntTensor(rois_num, output_dim, pooled_height, pooled_width)
        top_data = top_data.cuda()
        mapping_channes_data = mapping_channes_data.cuda()
        psroi_pooling.psroi_pooling_forward_cuda(pooled_height, pooled_width, spatial_scale, group_size, output_dim, input_data_tensor_trans, torch_roi_tensor, top_data, mapping_channes_data)
        output_result = top_data.permute(0, 2, 3, 1)
        output_node.setData(output_result.cpu())
        mapping_channes_result = mapping_channes_data.permute(0, 2, 3, 1)
        mapping_channes.setData(mapping_channes_result.cpu())

@registerProtoWriter("psroipool_forward")
class PsRoipoolProtoWriter(MluOpProtoWriter):
    def dumpOpParam2Node(self):
        param_node = self.proto_node_.psroipool_forward_param
        param_node.spatial_scale = self.op_params_.get("spatial_scale")
        param_node.group_size = self.op_params_.get("group_size")
        param_node.output_dim = self.op_params_.get("output_dim")
        param_node.pooled_height = self.op_params_.get("pooled_height")
        param_node.pooled_width = self.op_params_.get("pooled_width")

