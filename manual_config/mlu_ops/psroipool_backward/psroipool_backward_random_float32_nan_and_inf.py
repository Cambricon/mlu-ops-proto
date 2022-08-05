import sys
import os 
import numpy as np 
def dShape(shapes):
    shape_val = '"shape":['
    for i in range(len(shapes)-1):
        shape_val += str(shapes[i])+','
    shape_val += str(shapes[len(shapes)-1]) + ']'
    return  shape_val

def dType(data_type):
    return '"dtype":"' + data_type + '"'
def dRandomDistribution(start, end):
    return '"random_distribution":{"uniform":[' + str(start) + ',' + str(end) + ']}'
def dlayout(data_layout):
    return '"layout":"' + data_layout + '"'
def genSingleCase(dtype='float32', params_list=[1,1,2,2,1]):
    n = params_list[0]
    ci = params_list[1]
    h = params_list[2]
    w = params_list[3]
    rois_num = params_list[4]
    pooled_height = h
    pooled_width = w
    spatial_scale = format(np.random.random(), "0.5f")
    output_dim = ci

    bottom_c = ci * h * w
    bottom_b = np.random.randint(n, 2*n)
    bottom_h = np.random.randint(h, 10*h)
    bottom_w = np.random.randint(w, 10*w)

    top_grad_shape = [n, h, w, ci]
    roi_shape = [rois_num, 5]
    mappingChannel_shape = [n, h, w, ci]
    bottom_grad_shape = [bottom_b, bottom_h, bottom_w, bottom_c]
    inputs = '      {"inputs":['
    input1 = '{' + dShape(top_grad_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(-100,100) + "," +'"contain_nan":true,"contain_inf":true'+ ','+ dlayout("NHWC") + '}'
    input2 = '{' + dShape(mappingChannel_shape) + ',' + dType('int32') + ',' + dRandomDistribution(0,100) + "," + dlayout("NHWC") + '}'
    input3 = '{' + dShape(roi_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(0,200) + ',' + dlayout("ARRAY") + '}'
    outputs = '      "outputs":['
    output1 = '{' + dShape(bottom_grad_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(-50,50) + ',' + dlayout("NHWC") + '}'

    inputs += input1 + ',\n' + input2 + ',\n' + input3 +'],\n'
    outputs += output1 + '],\n'
    spatial_scale_s = '"spatial_scale": ' + str(spatial_scale)
    output_dim_s = '"output_dim": ' + str(output_dim)
    pooled_height_s = '"pooled_height": ' + str(pooled_height)
    pooled_width_s = '"pooled_width": ' + str(pooled_width)

    op_params = '     "op_params":{' + spatial_scale_s + ',' + output_dim_s + ', ' + pooled_height_s + ', '+ pooled_width_s + '},\n'

    proto_param = '     "proto_params":{"write_data":true}'

    cur_res = inputs + outputs + op_params + proto_param + '}'
    return cur_res

def genCase():
    cur_res = '     "manual_data":[\n'
    n = np.random.randint(1,5)
    ci = np.random.randint(1,5)
    h = np.random.randint(1,5)
    w = h
    num_rois = n
    param = [n ,ci, h, w, num_rois]
    cur_res += genSingleCase(params_list=param)
    for i in range(200):
        n = np.random.randint(1,5)
        ci = np.random.randint(1,10)
        h = np.random.randint(1,10)
        w = h
        num_rois = n
        param = [n ,ci, h, w, num_rois]
        cur_res += ',\n' + genSingleCase(params_list=param)
        if i % 6 ==0:
            n = np.random.randint(5,15)
            ci = np.random.randint(5,15)
            h = np.random.randint(5,15)
            w = h
            num_rois = n
            param = [n ,ci, h, w, num_rois]
            cur_res += ',\n' + genSingleCase(params_list=param)
            
        if i % 10 ==0:
            n = np.random.randint(10,30)
            ci = np.random.randint(10,30)
            h = np.random.randint(10,30)
            w = h
            num_rois = n
            param = [n ,ci, h, w, num_rois]
            cur_res += ',\n' + genSingleCase(params_list=param)
    cur_res += '\n      ]\n}'
    return cur_res

if __name__ == "__main__":
    res = '{\n\
    "op_name":"psroipool_backward",\n\
    "device":"gpu",\n\
    "data_type":{"input_dtype":["float32","int32","float32"], "output_dtype":["float32"]},\n\
    "random_distribution":{"uniform":[-100,100]},\n\
    "require_value":true,\n\
    "evaluation_criterion":["diff1","diff2"],\n\
    "evaluation_threshold":[0.003,0.003],\n'
    res += genCase()
    file = open("./psroipool_backward_manual_random_float32_nan_and_inf.json",'w')
    file.write(res)
    file.close()
