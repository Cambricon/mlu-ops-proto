import numpy as np
import torch
import poly_nms_cuda
import sys
import random

from nonmlu_ops.base import *

@registerTensorList('poly_nms')
class PolyNmsTensorList(TensorList):
    pass

@registerOp('poly_nms')
class PolyNmsOp(OpTest):
    def __init__(self, tensor_list, params):
        super().__init__(tensor_list, params)
        self.iou_threshold = self.params_.get("iou_threshold", 0.2)
        print("__init polynms")
        print(self.iou_threshold)

    def createCWCase(self):
        print("clock wise case.")
        np_boxes = self.tensor_list_.getInputTensor(0).getData()
        boxes = torch.from_numpy(np_boxes)
        up_bound = 10000000
        for i in range(boxes.shape[0]): 
            boxes[i][0] = random.uniform(0, up_bound)
            boxes[i][1] = random.uniform(0, up_bound)
            boxes[i][2] = random.uniform(0, up_bound)
            boxes[i][3] = random.uniform(0, -up_bound)
            boxes[i][4] = random.uniform(-up_bound, 0)
            boxes[i][5] = random.uniform(-up_bound, 0)
            boxes[i][6] = random.uniform(-up_bound, 0)
            boxes[i][7] = random.uniform(0, up_bound)
            boxes[i][8] = random.uniform(-up_bound, up_bound)
        return boxes
    
    def createCCWCase(self):
        print("counter clock wise case.")
        np_boxes = self.tensor_list_.getInputTensor(0).getData()
        boxes = torch.from_numpy(np_boxes)
        up_bound = 10000000
        for i in range(boxes.shape[0]): 
            boxes[i][0] = random.uniform(0, up_bound)
            boxes[i][1] = random.uniform(0, up_bound)
            boxes[i][2] = random.uniform(-up_bound, 0)
            boxes[i][3] = random.uniform(0, up_bound)
            boxes[i][4] = random.uniform(-up_bound, 0)
            boxes[i][5] = random.uniform(-up_bound, 0)
            boxes[i][6] = random.uniform(0, up_bound)
            boxes[i][7] = random.uniform(-up_bound, 0)
            boxes[i][8] = random.uniform(-up_bound, up_bound)
        return boxes

    def compute(self):
        dtype = self.tensor_list_.getInputTensor(0).getDataType()
        np_boxes = self.tensor_list_.getInputTensor(0).getData()
        iou_thresh = self.iou_threshold
        out_tensor = self.tensor_list_.getOutputTensor(0)
        out_tensor_length = self.tensor_list_.getOutputTensor(1)

        if iou_thresh * 10 % 2 == 0:
            boxes1 = self.createCWCase()
        else:
            boxes1 = self.createCCWCase()

        if (dtype == DataType.FLOAT32):
            boxes = boxes1.to(torch.float32).cuda()
            keep = poly_nms_cuda.poly_nms(boxes, iou_thresh)
            dims = out_tensor.getShape()[0]
            result = keep.cpu().numpy()
            real_dims = keep.shape[0]
            real_dims = dims - real_dims
            result = np.pad(result,(0,real_dims),'constant',constant_values=(0,0))
            out_tensor.setData(result)

            out_tensor_length.setData(keep.shape)
            print(out_tensor.getShape())
        else:
           raise Exception("poly_nms DataType should be Float, vs ", dtype)

@registerProtoWriter('poly_nms')
class OpTensorProtoWriter(MluOpProtoWriter):
    def dumpOpParam2Node(self):
        param_node = self.proto_node_.poly_nms_param
        param_node.iou_threshold = self.op_params_.get("iou_threshold", 0.1)
