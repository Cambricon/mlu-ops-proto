import numpy as np
import paddle.fluid as fluid
import paddle
import sys
import random
from nonmlu_ops.base import *

@registerTensorList('generate_proposals_v2')
class PolyNmsTensorList(TensorList):
    pass

@registerOp('generate_proposals_v2')
class PolyNmsOp(OpTest):
    def __init__(self, tensor_list, params):
        super().__init__(tensor_list, params)
        self.pre_nms_top_n = self.params_.get("pre_nms_top_n", 6000)
        self.post_nms_top_n = self.params_.get("post_nms_top_n", 2000)
        self.nms_thresh = self.params_.get("nms_thresh", 0.5)
        self.min_size = self.params_.get("min_size", 0.1)
        self.eta = self.params_.get("eta", 1.0)
        self.pixel_offset = self.params_.get("pixel_offset", True)
        print("__init generate_proposals_v2")

    def compute(self):
        paddle.disable_static()

        dtype = self.tensor_list_.getInputTensor(0).getDataType()
        np_scores = self.tensor_list_.getInputTensor(0).getData()
        np_bboox_deltas = self.tensor_list_.getInputTensor(1).getData()
        np_anchors = self.tensor_list_.getInputTensor(2).getData()
        np_variances = self.tensor_list_.getInputTensor(3).getData()
        np_img_size =  self.tensor_list_.getInputTensor(4).getData()

        N = np_scores.shape[0]
        H = np_scores.shape[1]
        W = np_scores.shape[2]
        A = np_scores.shape[3]

        # input is 【N，H，W, A】        
        np_scores=np_scores.transpose(0,3,1,2) # NHWA-->NAHW
        self.tensor_list_.getInputTensor(0).setData(np_scores)

        np_bboox_deltas = np_bboox_deltas.reshape(N, H ,W, A, 4) #[NHW4A]-->NA4HW
        np_bboox_deltas = np_bboox_deltas.transpose(0,3,4,1,2)
        np_bboox_deltas = np_bboox_deltas.reshape(N, A*4, H ,W)
        self.tensor_list_.getInputTensor(1).setData(np_bboox_deltas)

        np_rpn_rois = self.tensor_list_.getOutputTensor(0)
        np_rpn_roi_probs = self.tensor_list_.getOutputTensor(1)
        np_rpn_rois_num = self.tensor_list_.getOutputTensor(2)
        np_rpn_rois_batch_size = self.tensor_list_.getOutputTensor(3)

        if (dtype == DataType.FLOAT32):
            scores = paddle.to_tensor(np_scores, dtype=paddle.float32)
            bbox_deltas = paddle.to_tensor(np_bboox_deltas, dtype=paddle.float32)
            anchors = paddle.to_tensor(np_anchors, dtype=paddle.float32)
            variances = paddle.to_tensor(np_variances, dtype=paddle.float32)
            img_size = paddle.to_tensor(np_img_size, dtype=paddle.float32)
            rpn_rois, rpn_roi_probs, rpn_rois_num = paddle.vision.ops.generate_proposals(
                        scores,
                        bbox_deltas,
                        img_size,
                        anchors,
                        variances,
                        self.pre_nms_top_n,
                        self.post_nms_top_n,
                        self.nms_thresh,
                        self.min_size,
                        self.eta,
                        self.pixel_offset,
                        return_rois_num=True)
            rpn_rois_tmp = rpn_rois.numpy()
            print(rpn_rois.shape[0])
            print(self.post_nms_top_n)
            
            add_zero_count = N * self.post_nms_top_n - rpn_rois.shape[0]
            if(add_zero_count < 0):
                print('add zero count < 0')
            rpn_rois_tmp = rpn_rois_tmp.reshape(rpn_rois_tmp.shape[0]*rpn_rois_tmp.shape[1])
            rpn_rois_tmp = np.pad(rpn_rois_tmp,(0, add_zero_count*4),'constant',constant_values=(0,0))
            rpn_rois_tmp = rpn_rois_tmp.reshape(N * self.post_nms_top_n, 4)
            np_rpn_rois.setData(rpn_rois_tmp)

            print('add zero count:', add_zero_count)
            rpn_roi_probs_tmp = rpn_roi_probs.numpy()
            rpn_roi_probs_tmp = rpn_roi_probs_tmp.reshape(rpn_roi_probs_tmp.shape[0]*rpn_roi_probs_tmp.shape[1])
            rpn_roi_probs_tmp = np.pad(rpn_roi_probs_tmp,(0, add_zero_count),'constant',constant_values=(0,0))
            rpn_roi_probs_tmp = rpn_roi_probs_tmp.reshape(N * self.post_nms_top_n, 1)
            np_rpn_roi_probs.setData(rpn_roi_probs_tmp)


            np_rpn_rois_num.setData(rpn_rois_num.numpy())

            rpn_rois_batch_size_tmp = np.sum(rpn_rois_num.numpy())

            np_rpn_rois_batch_size.setData(np.array([rpn_rois_batch_size_tmp]))

            np_rpn_rois.setDiff(diff1=0.003, diff2=0.003, diff3=10.0)
            np_rpn_roi_probs.setDiff(diff1=10.0, diff2=10.0, diff3=0)
            np_rpn_rois_num.setDiff(diff1=10.0, diff2=10.0, diff3=0)
            np_rpn_rois_batch_size.setDiff(diff1=10.0, diff2=10.0, diff3=0)

            print("rpn_rois_batch_size", rpn_rois_batch_size_tmp)
            # input is 【N，H，W，A】  
            np_scores=np_scores.transpose(0,2,3,1)  # [N,A,H,W] ==> [N,H,W,A]
            self.tensor_list_.getInputTensor(0).setData(np_scores)

            np_bboox_deltas=np_bboox_deltas.reshape(N, A, 4, H ,W)
            np_bboox_deltas=np_bboox_deltas.transpose(0,3,4,1,2)
            np_bboox_deltas=np_bboox_deltas.reshape(N, H ,W,A*4)
            self.tensor_list_.getInputTensor(1).setData(np_bboox_deltas)

            print('N:', N)
            print('H:', H)
            print('W:', W)
            print('A:', A)

        else:
           raise Exception("generate_proposals_v2 DataType should be Float, vs ", dtype)

@registerProtoWriter('generate_proposals_v2')
class OpTensorProtoWriter(MluOpProtoWriter):
    def dumpOpParam2Node(self):
        param_node = self.proto_node_.generate_proposals_v2_param
        param_node.pre_nms_top_n = self.op_params_.get("pre_nms_top_n", 6000)
        param_node.post_nms_top_n = self.op_params_.get("post_nms_top_n", 2000)
        param_node.nms_thresh = self.op_params_.get("nms_thresh", 0.5)
        param_node.min_size = self.op_params_.get("min_size", 0.1)
        param_node.eta = self.op_params_.get("eta", 1.0)
        param_node.pixel_offset = self.op_params_.get("pixel_offset", True)
