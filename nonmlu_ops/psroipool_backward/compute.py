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
                elif (input_tensor == (self.input_tensors_)[1]):
                    mappingChannel_array = []
                    mappingChannels_shape = input_tensor.shape_
                    roi_num = mappingChannels_shape[0]
                    pooled_height = mappingChannels_shape[1]
                    pooled_width = mappingChannels_shape[2]
                    output_dim = mappingChannels_shape[3]
                    for b in range(0, roi_num):
                        for i in range(0, pooled_height * pooled_width):
                            for j in range(0, output_dim):
                                #value = h * pooled_width  + w + o * pooled_height * pooled_width
                                value = i + j * pooled_height * pooled_width
                                mappingChannel_array.append(value)
                    mappingChannels_narray = (np.array(mappingChannel_array)).reshape \
                                                (roi_num, pooled_height, pooled_width, output_dim)
                    input_tensor.getDataNode().setData(mappingChannels_narray)
                else:
                    RandomData(input_tensor).random()

@registerOp("psroipool_backward")
class PsRoiPooBackwardlOp(OpTest):
    def __init__(self, tensorlist, params):
        super().__init__(tensorlist, params)
        self.topGrad = self.tensor_list_.getInputTensor(0).getData()
        self.mappingChannel = self.tensor_list_.getInputTensor(1).getData()
        self.rois = self.tensor_list_.getInputTensor(2).getData()
        self.bottomGrad = self.tensor_list_.getOutputTensor(0).getData()
        self.topGrad_shape =  self.tensor_list_.getInputTensor(0).shape_
        self.mappingChannel_shape =  self.tensor_list_.getInputTensor(1).shape_
        self.roi_shape = self.tensor_list_.getInputTensor(2).shape_
        self.bottomGrad_shape = self.tensor_list_.getOutputTensor(0).shape_
        self.spatial_scale = self.params_.get("spatial_scale")
        self.output_dim = self.params_.get("output_dim")
        self.pooled_height = self.params_.get("pooled_height")
        self.pooled_width = self.params_.get("pooled_width")

    def compute(self):
        # param check
        bottomGrad_shape = self.bottomGrad_shape
        batch = bottomGrad_shape[0]
        height = bottomGrad_shape[1]
        width = bottomGrad_shape[2]
        channel = bottomGrad_shape[3]
        roi_shape = self.roi_shape
        rois_num = roi_shape[0]
        rois_offset = roi_shape[1]
        spatial_scale = self.spatial_scale
        output_dim = self.output_dim
        pooled_height = self.pooled_height
        pooled_width = self.pooled_width
        if (channel != pooled_height * pooled_width * output_dim):
            print("Error: the channel must be equal to pooled_height * pooled_width * output_dim, channel is = %d, pooled_height is %d, pooled_width is %d,output_dim is %d" %(channel, pooled_height, pooled_width, output_dim))
        if rois_offset != 5:
            print("Error: the roi input second dim must be equal to 5, but now is %d" %rois_offset)
        if (pooled_width != pooled_height):
            print("Error: the pooled_width must be equal to pooled_height, but now pooled_width is %d, pooled_height is %d" %(pooled_width, pooled_height))
        if (len(self.tensor_list_.getInputTensors()) != 3):
            print("Error: the input tensors must be equal to 3, but now is %d" %(len(self.tensor_list_.getInputTensors())))
        if (len(self.tensor_list_.getOutputTensors()) != 1):
            print("Error: the output tensors must be equal to 1, but now is %d" %(len(self.tensor_list_.getOutputTensors())))

        inputs : list[Tensor] =  self.tensor_list_.getInputTensors()
        topGrad_layout = inputs[0].layout_
        mappingChannel_layout = inputs[1].layout_
        bottomGrad_node = self.tensor_list_.getOutputTensor(0)
        bottomGrad_layout = bottomGrad_node.layout_
        
        if (topGrad_layout != Layout.NHWC):
            print("ERROR: the input data layout only support NHWC, but now is " + topGrad_layout)
        if (mappingChannel_layout != Layout.NHWC):
            print("ERROR: the input data layout only support NHWC, but now is " + mappingChannels_layout)
        if (bottomGrad_layout != Layout.NHWC):
            print("ERROR: the output data layout only support NHWC, but now is " + bottomGrad_layout)
        topGrad_tensor = torch.from_numpy(self.topGrad)
        mappingChannel_tensor = torch.from_numpy(self.mappingChannel)
        torch_roi_tensor = torch.from_numpy(self.rois).cuda()

         # set rois value
        roi_value = []
        for roi_id in range(0, int(rois_num)):
            roi_tmp = []
            x_a = random.random() * (float(width) / spatial_scale)
            x_b = random.random() * (float(width) / spatial_scale)
            roi_x_start = x_a if x_a < x_b else x_b
            roi_x_end = x_a if x_a > x_b else x_b
            y_a = random.random() * (float(height) / spatial_scale)
            y_b = random.random() * (float(height) / spatial_scale)
            roi_y_start = y_a if y_a < y_b else y_b
            roi_y_end = y_a if y_a > y_b else y_b

            roi_tmp.append((int)(random.randint(0, batch - 1)))
            roi_tmp.append(format(roi_x_start, "0.5f"))
            roi_tmp.append(format(roi_y_start, "0.5f"))
            roi_tmp.append(format(roi_x_end, "0.5f"))
            roi_tmp.append(format(roi_y_end, "0.5f"))
            roi_value.extend(roi_tmp)

        roi_array = (np.array(roi_value)).reshape(rois_num, rois_offset)
        roi_node = self.tensor_list_.getInputTensor(2)
        roi_node.setData(roi_array)

        # the input data must be trans to NCHW
        topGrad_tensor_trans = topGrad_tensor.permute(0, 3, 1, 2).cuda()
        mappingChannel_tensor_trans = mappingChannel_tensor.permute(0, 3, 1, 2).cuda()
        bottomGrad_data = torch.zeros(batch, channel, height, width).cuda()
        psroi_pooling.psroi_pooling_backward_cuda(pooled_height, pooled_width, spatial_scale, output_dim, topGrad_tensor_trans, torch_roi_tensor, bottomGrad_data, mappingChannel_tensor_trans)
        bottomGrad_result = bottomGrad_data.permute(0, 2, 3, 1)
        bottomGrad_node.setData(bottomGrad_result.cpu())

@registerProtoWriter("psroipool_backward")
class PsRoiPoolBackwardProtoWriter(MluOpProtoWriter):
    def dumpOpParam2Node(self):
        param_node = self.proto_node_.psroipool_backward_param
        param_node.spatial_scale = self.op_params_.get("spatial_scale")
        param_node.output_dim = self.op_params_.get("output_dim")
        param_node.pooled_height = self.op_params_.get("pooled_height")
        param_node.pooled_width = self.op_params_.get("pooled_width")