from math import fabs
import paddle
import paddle.fluid as fluid
import numpy as np
from nonmlu_ops.base import *
#paddle.enable_static()

@registerTensorList("prior_box")
class PriorBoxTensorList(TensorList):
    pass

@registerOp("prior_box")
class PriorBoxOp(OpTest):
    def __init__(self, tensorlist, params):
        super().__init__(tensorlist, params)
        # get input tensor
        self.min_sizes_ = self.tensor_list_.getInputTensor(0)
        self.aspect_ratios_ = self.tensor_list_.getInputTensor(1)
        self.variances_ = self.tensor_list_.getInputTensor(2)
        self.max_sizes_ = self.tensor_list_.getInputTensor(3)
        # get op params
        self.height_ = self.params_.get("height")
        self.width_ = self.params_.get("width")
        self.im_height_ = self.params_.get("im_height")
        self.im_width_ = self.params_.get("im_width")
        self.flip_ = self.params_.get("flip")
        self.clip_ = self.params_.get("clip")
        self.step_w_ = self.params_.get("step_w")
        self.step_h_ = self.params_.get("step_h")
        self.offset_ = self.params_.get("offset")
        self.min_max_aspect_ratios_order_ = self.params_.get("min_max_aspect_ratios_order")
        # get output tensor 
        self.output_ = self.tensor_list_.getOutputTensor(0)
        self.var_ = self.tensor_list_.getOutputTensor(1)
    
    def generator_new_aspect(self,flip):
        new_aspect_ratios = []
        epsilon = 1e-6
        aspect_ratios = self.aspect_ratios_.getData().tolist()
        new_aspect_ratios.clear()
        new_aspect_ratios.append(1.0)
        for ar in aspect_ratios:
            already_exist = False
            for new_ar in new_aspect_ratios:
                if fabs(ar-new_ar) < epsilon:
                    already_exist = True
                    break
            if not already_exist:
                new_aspect_ratios.append(ar)
                if flip:
                    new_aspect_ratios.append(1.0 / ar)
        self.aspect_ratios_.setShape([len(new_aspect_ratios)])
        self.aspect_ratios_.getDataNode().setData(np.array(new_aspect_ratios))

    def compute(self):
        input_shape = [1,1,self.height_,self.width_]
        paddle_input = paddle.ones(input_shape)
        image_shape = [1,1,self.im_height_,self.im_width_]
        paddle_image = paddle.ones(image_shape)

        min_sizes = self.min_sizes_.getData()
        paddle_min_sizes = min_sizes.tolist()
        aspect_ratios = self.aspect_ratios_.getData()
        paddle_aspect_ratios = aspect_ratios.tolist()
        variances = self.variances_.getData()
        paddle_variances = variances.tolist()
        max_sizes = self.max_sizes_.getData()
        paddle_max_sizes = max_sizes.tolist()
        paddle_flip = self.flip_
        paddle_clip = self.clip_
        paddle_min_max_aspect_ratios_order = self.min_max_aspect_ratios_order_
        paddle_step_w = self.step_w_
        paddle_step_h = self.step_h_
        paddle_offset = self.offset_
        assert (paddle_step_h > 0) and (paddle_step_w > 0), "ERROR:the input data step_h/step_w must greater than zero." 
        if paddle_max_sizes:
            assert len(paddle_min_sizes) == len(paddle_max_sizes), "ERROR:the input data min_sizes \
                and max_sizes length must be equal" 
            assert [paddle_min_sizes[i] < paddle_max_sizes[i] for i in range(len(paddle_min_sizes))],\
                "Error:the input data min_sizes[i] must be smaller than max_sizes[i]"
        assert len(paddle_variances)==4, "ERROR:the input data variances length must be equal 4."
        box, var = fluid.layers.prior_box(input=paddle_input,
                                          image=paddle_image,
                                          min_sizes=paddle_min_sizes,
                                          max_sizes=paddle_max_sizes,
                                          aspect_ratios=paddle_aspect_ratios,
                                          variance = paddle_variances,
                                          flip=paddle_flip,
                                          clip=paddle_clip,
                                          steps=[float(paddle_step_w),float(paddle_step_h)],
                                          offset=float(paddle_offset),
                                          min_max_aspect_ratios_order=paddle_min_max_aspect_ratios_order)
        num_priors = self.output_.getShape()[2]
        print("the out num_priors = ",num_priors)
        self.generator_new_aspect(self.flip_)
        new_aspect_size = self.aspect_ratios_.getData().tolist()
        new_num_priors = 0
        new_num_priors = len(paddle_min_sizes) * len(new_aspect_size)
        if paddle_max_sizes:
             new_num_priors += len(paddle_max_sizes)
        if num_priors != new_num_priors:
            print("WARNING: the output shape set in json is inaccurate ,we change the shape in this place")
            self.output_.setShape([self.height_,self.width_,new_num_priors,4])
            self.var_.setShape([self.height_,self.width_,new_num_priors,4])
        print("paddle_box/var_shape:",box.shape)
        print("mlu_box/var_shape:",self.output_.getShape())
        assert box.shape == self.output_.getShape(),"ERROR: the mlu_output shape not equal paddle output shape."
        self.output_.getDataNode().setData(box.numpy())
        self.output_.setDiff(diff1=0.003,diff2=0.003,diff3=1e+6)
        self.var_.getDataNode().setData(var.numpy())
        self.var_.setDiff(diff1=1e+6,diff2=1e+6,diff3=0)

@registerProtoWriter("prior_box")
class PriorBoxProtoWriter(MluOpProtoWriter):
    def dumpOpParam2Node(self):
        param_node = self.proto_node_.prior_box_param
        param_node.height = self.op_params_.get("height")
        param_node.width = self.op_params_.get("width")
        param_node.im_height = self.op_params_.get("im_height")
        param_node.im_width = self.op_params_.get("im_width")
        param_node.flip = self.op_params_.get("flip")
        param_node.clip = self.op_params_.get("clip")
        param_node.step_w = self.op_params_.get("step_w")
        param_node.step_h = self.op_params_.get("step_h")
        param_node.offset = self.op_params_.get("offset")
        param_node.min_max_aspect_ratios_order = self.op_params_.get("min_max_aspect_ratios_order")
        