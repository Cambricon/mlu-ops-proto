from nonmlu_ops.base import *
import tensorflow.compat.v1 as tf
from tensorflow.python.training import gen_training_ops

@registerTensorList("expand")
class ExpandTensorList(TensorList):
    pass

@registerOp("expand")
class ExpandOp(OpTest):
    def __init__(self, tensorlist, params):
        super().__init__(tensorlist, params)
        self.input_ = self.tensor_list_.getInputTensor(0).getShape()
        self.output_ = self.tensor_list_.getOutputTensor(0).getShape()
        self.params_ = self.params_.get("params",[])
        self.computeOutputShape()

    def computeOutputShape(self):
        self.tensor_list_.getOutputTensor(0).setShape(self.output_)

    def compute(self):
        input_tensor = self.tensor_list_.getInputTensor(0)
        output_tensor = self.tensor_list_.getOutputTensor(0)
        # compute baseline output
        try:
            init = tf.global_variables_initializer()
            if input_tensor.getDataType().isComplex():
                real_data, imag_data = input_tensor.getDataNode().getComplexData()
                complex_data = np.array(real_data, dtype=np.complex128)
                complex_data.imag = imag_data
                expand_op = tf.broadcast_to(complex_data, self.params_)
                with tf.Session() as sess:
                    sess.run(init)
                    result = sess.run(expand_op)
                    output_tensor.setComplexData(result.real, result.imag)
            else:
                input1 = input_tensor.getDataNode()
                tf_input1 = tf.placeholder(input1.dtype_.getNumpyStr())  
                expand_op = tf.broadcast_to(tf_input1, self.params_)
                with tf.Session() as sess:
                    sess.run(init)
                    result = sess.run(expand_op, feed_dict={tf_input1:input1.getData()})
                    output_tensor.setData(result)
        except ImportError:
            one_array = np.ones(out_tensor.shape_)
            out_tensor.setData(one_array)
        
@registerProtoWriter("expand")
class ExpandProtoWriter(MluOpProtoWriter):
   pass 
