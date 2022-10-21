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
def genSingleCase(dtype='float32', params_list=[1,1,1,1]):
    b = params_list[0]
    c = params_list[1]
    m = params_list[2]
    n = params_list[3]

    
    input_shape_0 = [b, c, m]
    input_shape_1 = [b, n, 3]
    input_shape_2 = [b, n, 3]
    inputs = '     {\n       "inputs":['
    input0 = '{' + dShape(input_shape_0) + ',' + dRandomDistribution(-1,1) + "," + dlayout("ARRAY") + '}'
    input1 = '{' + dShape(input_shape_1) + ',' + dRandomDistribution(0,m) + "," + dlayout("ARRAY") + '}'
    input2 = '{' + dShape(input_shape_2) + ',' + dRandomDistribution(-1,1) + "," + dlayout("ARRAY") + '}'
    
    output_shape_0 = [b, c, n]
    outputs = '       "outputs":['
    output0 = '{' + dShape(output_shape_0) + ',' + dlayout("ARRAY") + '}'


    inputs += input0 + ',\n                 ' +  input1 + ',\n                 ' +  input2 + '],\n'
    outputs += output0 + ']'
    
    cur_res = inputs + outputs + '\n     }'
    return cur_res

def genCase():
    cur_res = '     "manual_data":[\n'
    # b_list = [1,2,4,8,16,24,32,64,128,256,512]
    # cmn_list = [[512, 16, 64], [256, 64, 256], [256, 256, 1024], [128, 1024, 4096], [16, 512, 64], [64, 246, 256], [1024, 128, 4096], [1, 1024, 128], [128, 256, 512], [512, 128, 1024], [512, 128, 2048]]
    b_list = [1,2,4,8,16,24,32,64,128,256,512]
    cmn_list=[[1, 1, 100], [1, 2, 100], [1, 3, 100]]

    k = 0
    for b in b_list:
        for cmn in cmn_list:
            param = [b, cmn[0], cmn[1], cmn[2]]
            if 0 == k:
                cur_res += genSingleCase(params_list=param)
            else:
                cur_res += ',\n' + genSingleCase(params_list=param)
            k = k + 1

    cur_res += '\n     ]\n}'
    return cur_res

if __name__ == "__main__":
    res = '{\n\
    "op_name":"three_interprolate_forward",\n\
    "device":"gpu",\n\
    "require_value":true,\n\
    "data_type":{"input_dtype":["float32", "int32", "float32"], "output_dtype":["float32"]},\n\
    "evaluation_criterion":["diff1", "diff2"],\n\
    "evaluation_threshold":[3e-3, 3e-3],\n'
    res += genCase()
    file = open("./manual_net_half_specail_0.json",'w')
    file.write(res)
    file.close()