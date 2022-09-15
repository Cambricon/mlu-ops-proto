import paddle
import numpy as np
import sys

from nonmlu_ops.base import *

@registerTensorList('yolo_box')
class YoloBoxTensorList(TensorList):
    pass

@registerOp('yolo_box')
class YoloBoxOp(OpTest):
    def __init__(self, tensor_list, params):
        super().__init__(tensor_list, params)
        self.class_num = self.params_.get("class_num", 1)
        self.conf_thresh = self.params_.get("conf_thresh", 0.01)
        self.downsample_ratio = self.params_.get("downsample_ratio", 8)
        self.clip_bbox = self.params_.get("clip_bbox", True)
        self.scale_x_y = self.params_.get("scale_x_y", 1.0)
        self.iou_aware = self.params_.get("iou_aware", True)
        self.iou_aware_factor = self.params_.get("iou_aware_factor", 0.5)
        self.x_shape = self.tensor_list_.getInputTensor(0).shape_

    def compute(self):
        # input tensor
        x_tensor = self.tensor_list_.getInputTensor(0).getData()
        img_size_tensor = self.tensor_list_.getInputTensor(1).getData()
        anchors_tensor = self.tensor_list_.getInputTensor(2).getData()

        # output tensor
        boxes_tensor = self.tensor_list_.getOutputTensor(0)
        scores_tensor = self.tensor_list_.getOutputTensor(1)

        x = paddle.to_tensor(x_tensor)
        img_size = paddle.to_tensor(img_size_tensor)
        anchors = anchors_tensor.tolist()
    
        s = int(len(anchors)/2)
        n = int(self.x_shape[0])
        h = int(self.x_shape[2])
        w = int(self.x_shape[3])

        boxes, scores = paddle.vision.ops.yolo_box(x,
                                                   img_size=img_size,
                                                   anchors=anchors,
                                                   class_num=self.class_num,
                                                   conf_thresh=self.conf_thresh,
                                                   downsample_ratio=self.downsample_ratio,
                                                   clip_bbox=self.clip_bbox,
                                                   scale_x_y=self.scale_x_y,
                                                   iou_aware=self.iou_aware,
                                                   iou_aware_factor=self.iou_aware_factor)
        
        boxes = boxes.reshape([n, s, h*w, 4]).transpose([0, 1, 3, 2])
        scores = scores.reshape([n, s, h*w, self.class_num]).transpose([0, 1, 3, 2])
        boxes_tensor.setData(boxes.cpu())
        scores_tensor.setData(scores.cpu())

@registerProtoWriter('yolo_box')
class OpTensorProtoWriter(MluOpProtoWriter):
    def dumpOpParam2Node(self):
        param_node = self.proto_node_.yolo_box_param
        param_node.class_num = self.op_params_.get("class_num", 1)
        param_node.conf_thresh = self.op_params_.get("conf_thresh", 0.01)
        param_node.downsample_ratio = self.op_params_.get("downsample_ratio", 8)
        param_node.clip_bbox = self.op_params_.get("clip_bbox", True)
        param_node.scale_x_y = self.op_params_.get("scale_x_y", 1.0)
        param_node.iou_aware = self.op_params_.get("iou_aware", True)
        param_node.iou_aware_factor = self.op_params_.get("iou_aware_factor", 0.5)
