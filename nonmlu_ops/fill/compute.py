from nonmlu_ops.base import *
import torch
import numpy as np
import tensorflow as tf

@registerTensorList("fill")
class FillTensorList(TensorList):
    pass

@registerOp("fill")
class FillOp(OpTest):
    def __init__(self, tensorlist, params):
        super().__init__(tensorlist, params)
        self.version_ = self.params_.get("version")
        self.mode_ = self.params_.get("mode")
        if "value" in self.params_:
            self.value_ = self.params_.get("value")
        elif "value_hex" in self.params_:
            value = self.params_.get("value_hex")
            self.value_ = int(value, 16)

    def compute(self):
        input_tensor = self.tensor_list_.getInputTensor(0)
        input_dtype = input_tensor.getDataType().getStr()
        #pytorch does not support uint16/32/64
        if (input_dtype in ("uint16", "uint32", "uint64")):
            #since tensorflow does not support strides, 
            #I convert all uint16/32/64 dtypes to int16/32/64 in pytorch,
            #so when input_dtype is uint16/32/64, the generated case cannot reach the boundary value
            if input_tensor.strides_ is not None:
                if input_dtype == "uint16":
                    input_data = torch.tensor(input_tensor.getData().astype(np.int16)).cuda(0)
                elif input_dtype == "uint32":
                    input_data = torch.tensor(input_tensor.getData().astype(np.int32)).cuda(0)
                elif input_dtype == "uint64":
                    input_data = torch.tensor(input_tensor.getData().astype(np.int64)).cuda(0) 
                input_data = torch.as_strided(input_data, input_tensor.shape_, input_tensor.strides_)
                result = (torch.fill_(input_data, self.value_)).cpu().numpy()
            else:
                init = tf.compat.v1.global_variables_initializer()
                input_data = input_tensor.getData()
                out_fill = (tf.fill(input_data.shape, self.value_))
                config = tf.compat.v1.ConfigProto()
                config.gpu_options.allow_growth = True
                with tf.compat.v1.Session(config=config) as sess:
                    sess.run(init)
                    result = sess.run(out_fill)
        if (input_dtype in ("bool", "int8", "int16", "int32", "int64", "uint8", "float16", "float32")):
            if input_tensor.strides_ is not None:
                input_data = torch.tensor(input_tensor.getData()).cuda(0)
                input_data = torch.as_strided(input_data, input_tensor.shape_, input_tensor.strides_)
            else:
                input_data = torch.tensor(input_tensor.getData()).cuda(0)
            result = (torch.fill_(input_data, self.value_)).cpu().numpy()
        output_tensor = self.tensor_list_.getOutputTensor(0)
        output_tensor.setShape(result.shape)
        output_tensor.setData(result)
        if input_tensor.strides_ is not None:
            output_tensor.strides_ = input_tensor.strides_

@registerProtoWriter("fill")
class FillProtoWriter(MluOpProtoWriter):
    def dumpOpParam2Node(self):
        fill_param_node = self.proto_node_.fill_param
        if "value" in self.op_params_:
            fill_param_node.value = self.op_params_.get("value")
        fill_param_node.version = self.op_params_.get("version")
        fill_param_node.mode = self.op_params_.get("mode")
        if "value_hex" in self.op_params_:
            fill_param_node.value_hex = self.op_params_.get("value_hex")

