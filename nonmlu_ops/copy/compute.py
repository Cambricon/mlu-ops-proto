import sys
import tensorflow as tf 
import numpy as np
from nonmlu_ops.base import *

@registerTensorList("copy")
class CopyTensorList(TensorList):
    pass

@registerOp("copy")
class CopyOp(OpTest):
    #Initialize parameters and tensor
    def __init__(self, tensorlist, params):
        print(tf.__version__)
        super().__init__(tensorlist, params)

    def compute(self):
        input_tensor = self.tensor_list_.getInputTensor(0)
        output_tensor = self.tensor_list_.getOutputTensor(0)

        assert(input_tensor.getDataType() == output_tensor.getDataType())

        if input_tensor.getDataType().isComplex():
            real_data, imag_data = input_tensor.getDataNode().getComplexData()
            cpu_dtype = real_data.dtype
            # since tensorflow does not support complex_half, 
            # we convert all complex dtype to complex_double
            # this is OK for pure IO operators, such as strided_slice, copy etc.
            cmplx_data = np.array(real_data, dtype=np.complex128)
            cmplx_data.imag = imag_data
            x = tf.compat.v1.placeholder(dtype=tf.complex128, shape=input_tensor.getShape())
            op = tf.raw_ops.Snapshot(input=x)
            with tf.compat.v1.Session() as sess:
                output_data = sess.run(op, feed_dict={x: cmplx_data})
            output_tensor.setShape(output_data.shape)
            # convert double dtype back to input cpu_dtype
            cpu_real_data = np.array(output_data.real, dtype=cpu_dtype)
            cpu_imag_data = np.array(output_data.imag, dtype=cpu_dtype)
            output_tensor.setComplexData(cpu_real_data, cpu_imag_data)
        else:
            input_data = input_tensor.getDataNode().getData()
            x = tf.compat.v1.placeholder(dtype=input_tensor.getDataType().getStr(), shape=input_tensor.getShape())
            op = tf.raw_ops.Snapshot(input=x)
            with tf.compat.v1.Session() as sess:
                output_data = sess.run(op, feed_dict={x: input_data})
            output_tensor.setShape(output_data.shape)
            output_tensor.setData(output_data)

@registerProtoWriter("copy")
class CopyProtoWriter(MluOpProtoWriter):
    pass
