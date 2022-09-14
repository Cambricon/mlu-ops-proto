import numpy as np
import torch 
from nonmlu_ops.base import *

# from mmcv.ops import ball_query
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext('_ext', ['ball_query_forward'])

@registerTensorList("ball_query")
class BallQueryTensorList(TensorList):
  pass

@registerOp("ball_query")
class BallQueryOp(OpTest):
  def __init__(self, tensor_list, param):
    super().__init__(tensor_list, param)
    self.new_xyz = self.tensor_list_.getInputTensor(0)
    self.xyz     = self.tensor_list_.getInputTensor(1)
    self.out_tensor  = self.tensor_list_.getOutputTensor(0) 
    self.min_radius = self.params_.get("min_radius") 
    self.max_radius = self.params_.get("min_radius")
    self.nsample    = self.params_.get("nsample")

  def compute(self):
    # param check
    new_xyz_shape = self.new_xyz.shape_
    batch_new_xyz = new_xyz_shape[0]
    num_new_xyz   = new_xyz_shape[1]
    channel_new_xyz = new_xyz_shape[2]
    xyz_shape     = self.xyz.shape_
    batch_xyz     = xyz_shape[0]
    num_xyz       = xyz_shape[1]
    channel_xyz   = xyz_shape[2]
    min_radius    = self.min_radius
    max_radius    = self.max_radius
    nsample       = self.nsample
    assert (batch_new_xyz == batch_xyz)
    assert (num_new_xyz <= num_xyz)
    assert (channel_new_xyz == 3 and channel_new_xyz == channel_xyz)
    assert (min_radius >= 0 and max_radius >= 0 and min_radius <= max_radius)
    assert (nsample <= num_xyz and nsample >= 1)

    new_xyz_dtype = self.new_xyz.getDataType()
    xyz_dtype     = self.xyz.getDataType()
    out_dtype     = self.out_tensor.getDataType()
    assert (new_xyz_dtype == xyz_dtype)
    assert (new_xyz_dtype == DataType.FLOAT32 or new_xyz_dtype == DataType.FLOAT16)
    assert (out_dtype == DataType.INT32)

    new_xyz_tensor = torch.from_numpy(self.new_xyz.getData()).cuda()
    xyz_tensor     = torch.from_numpy(self.xyz.getData()).cuda()
    idx = xyz_tensor.new_zeros(batch_new_xyz, num_new_xyz, nsample, dtype=torch.int)
    ext_module.ball_query_forward(
        new_xyz_tensor,
        xyz_tensor,
        idx,
        b=batch_new_xyz,
        n=num_xyz,
        m=num_new_xyz,
        min_radius=min_radius,
        max_radius=max_radius,
        nsample=nsample)
    self.out_tensor.setData(idx.cpu())

@registerProtoWriter("ball_query")
class BallQueryProtoWriter(MluOpProtoWriter):
  def dumpOpParam2Node(self):
    param_node = self.proto_node_.ball_query_param
    param_node.min_radius = self.op_params_.get("min_radius")
    param_node.max_radius = self.op_params_.get("max_radius")
    param_node.nsample    = self.op_params_.get("nsample")