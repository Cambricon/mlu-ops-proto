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

    def createRandomOrderCase(self):
        distribution, random_range = next(iter(self.tensor_list_.getInputTensor(0).random_distribution_.items()))
        start = random_range[0]
        end = random_range[1]
        width = (end - start) / 2
        mid = start + width

        np_boxes = self.tensor_list_.getInputTensor(0).getData()
        boxes = torch.from_numpy(np_boxes)
        for i in range(boxes.shape[0]):
            if i % 2 == 0:
                # print("clock wise case.")
                boxes[i][0] = random.uniform(mid, end)
                boxes[i][1] = random.uniform(mid, end)

                boxes[i][2] = random.uniform(mid, end)
                boxes[i][3] = random.uniform(start, mid)

                boxes[i][4] = random.uniform(start, mid)
                boxes[i][5] = random.uniform(start, mid)

                boxes[i][6] = random.uniform(start, mid)
                boxes[i][7] = random.uniform(mid, end)

                boxes[i][8] = random.uniform(start, end)
            else:
                # print("counter clock wise case.")
                boxes[i][0] = random.uniform(mid, end)
                boxes[i][1] = random.uniform(mid, end)

                boxes[i][2] = random.uniform(start, mid)
                boxes[i][3] = random.uniform(mid, end)

                boxes[i][4] = random.uniform(start, mid)
                boxes[i][5] = random.uniform(start, mid)

                boxes[i][6] = random.uniform(mid, end)
                boxes[i][7] = random.uniform(start, mid)

                boxes[i][8] = random.uniform(start, end)
        return boxes

    def compute(self):
        dtype = self.tensor_list_.getInputTensor(0).getDataType()
        iou_thresh = self.iou_threshold
        out_tensor = self.tensor_list_.getOutputTensor(0)
        out_tensor_length = self.tensor_list_.getOutputTensor(1)

        boxes_data = self.createRandomOrderCase()

        if (dtype == DataType.FLOAT32):
            boxes = boxes_data.to(torch.float32).cuda()
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
