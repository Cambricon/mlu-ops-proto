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
    pooled_height = np.random.randint(1,max(1.2*h, 2))
    pooled_width = np.random.randint(1,max(1.2*w, 2))

    gradInput_shape = [n, h, w, ci]
    grid_shape = [rois_num*n,pooled_height,pooled_width,2]
    gradOutput_shape = [rois_num*n, pooled_height, pooled_width, ci]
    inputs = '      {"inputs":['
    input1 = '{' + dShape(gradOutput_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(-50,50) + "," +'"contain_nan":true,"contain_inf":true'+ ','+ dlayout("NHWC") + '}'
    input2 = '{' + dShape(grid_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(-1,1) + ',' + dlayout("ARRAY") + '}'
    outputs = '      "outputs":['
    output1 = '{' + dShape(gradInput_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(-50,50) + ',' + dlayout("NHWC") + '}'

    inputs += input1 + ',\n' + input2 + '],\n'
    outputs += output1 + '],\n'
    
    proto_param = '     "proto_params":{"write_data":true}'

    cur_res = inputs + outputs + proto_param + '}'
    return cur_res

def genCase():
    cur_res = '     "manual_data":[\n'
    n = np.random.randint(1,10)
    ci = np.random.randint(1,10)
    h = np.random.randint(1,10)
    w = np.random.randint(1,10)
    num_rois = np.random.randint(1,5)
    param = [n ,ci, h, w, num_rois]
    cur_res += genSingleCase(params_list=param)
    for i in range(10):
        n = np.random.randint(1,10)
        ci = np.random.randint(5,10)
        h = np.random.randint(4,20)
        w = np.random.randint(4,20)
        num_rois = np.random.randint(1,10)
        param = [n ,ci, h, w, num_rois]
        cur_res += ',\n' + genSingleCase(params_list=param)
        if i % 3 ==0:
            n = np.random.randint(10,40)
            ci = np.random.randint(10,40)
            h = np.random.randint(20,40)
            w = np.random.randint(20,40)
            num_rois = np.random.randint(2,7)
            param = [n ,ci, h, w, num_rois]
            cur_res += ',\n' + genSingleCase(params_list=param)
            
        if i%5 ==0:
            n = np.random.randint(50,100)
            ci = np.random.randint(100,300)
            h = np.random.randint(5,30)
            w = np.random.randint(5,30)
            num_rois = np.random.randint(1,5)
            param = [n ,ci, h, w, num_rois]
            cur_res += ',\n' + genSingleCase(params_list=param)
    cur_res += '\n      ]\n}'
    return cur_res

if __name__ == "__main__":
    res = '{\n\
    "op_name":"roi_crop_backward",\n\
    "device":"gpu",\n\
    "data_type":{"input_dtype":["float32","float32"], "output_dtype":["float32"]},\n\
    "random_distribution":{"uniform":[-10,10]},\n\
    "require_value":true,\n\
    "evaluation_criterion":["diff1","diff2"],\n\
    "evaluation_threshold":[0.003,0.003],\n'
    res += genCase()
    file = open("./roi_crop_backward_manual_random_float32_nan_and_inf.json",'w')
    file.write(res)
    file.close()

