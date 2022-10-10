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
def genSingleCase(dtype='float32', params_list=[1,1,1,1,False]):
    B = params_list[0]
    C = params_list[1]
    M = params_list[2]
    N = params_list[3]
    nan_inf = params_list[4]

    features_shape = [B, C, M]
    idx_shape     = [B, N, 3]
    weight_shape     = [B, N, 3]
    output_shape = [B, C, N]

    inputs = '     {\n       "inputs":['
    if nan_inf:
      input1 = '{' + dShape(features_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(0,100) + ","+'"contain_nan":true,"contain_inf":true'+ "," + dlayout("ARRAY") + '}'
      input2 = '{' + dShape(idx_shape) + ',' + dType('int32') + ',' + dRandomDistribution(0,M-1) + ","+'"contain_nan":true,"contain_inf":true'+ "," + dlayout("ARRAY") + '}'
      input3 = '{' + dShape(weight_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(0,1) + ","+'"contain_nan":true,"contain_inf":true'+ "," + dlayout("ARRAY") + '}'
    else :
      input1 = '{' + dShape(features_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(0,100) + "," + dlayout("ARRAY") + '}'
      input2 = '{' + dShape(idx_shape) + ',' + dType('int32') + ',' + dRandomDistribution(0,M-1) + "," + dlayout("ARRAY") + '}'
      input3 = '{' + dShape(weight_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(0,1) + "," + dlayout("ARRAY") + '}'

    outputs = '       "outputs":['
    output1 = '{' + dShape(output_shape) + ',' + dType(dtype) + ',' + dlayout("ARRAY") + '}'


    inputs += input1 + ',\n' + '                 ' + input2 + ',\n' + '                 ' + input3 + '],\n'
    outputs += output1  + '],\n'

    proto_param = '       "proto_params":{"write_data":true}'

    cur_res = inputs + outputs + proto_param + '\n     }'
    return cur_res

def genCase(dtype = "float32", nan_inf=False):
    count = 1
    cur_res = '     "manual_data":[\n'
    B = np.random.randint(1,32)
    C = np.random.randint(1,128)
    M = np.random.randint(1,128)
    N = np.random.randint(1,128)
    param = [B, C, M, N, nan_inf]
    cur_res += genSingleCase(dtype = dtype, params_list=param)

    count += 1
    B = np.random.randint(1,32)
    C = np.random.randint(129,256)
    M = np.random.randint(129,256)
    N = np.random.randint(129,256)
    param = [B, C, M, N, nan_inf]
    cur_res += ',\n' + genSingleCase(dtype = dtype, params_list=param)
 
    for i in range(5):
        if i % 2 == 0:
            count += 1
            B = np.random.randint(1,32)
            C = np.random.randint(257,512)
            M = np.random.randint(257,512)
            N = np.random.randint(257,512)
            param = [B, C, M, N, nan_inf]
            cur_res += ',\n' + genSingleCase(dtype = dtype, params_list=param)

        if i % 3 == 0:
            count += 1
            B = np.random.randint(1,32)
            C = np.random.randint(513,1024)
            M = np.random.randint(513,1024)
            N = np.random.randint(513,1024)
            param = [B, C, M, N, nan_inf]
            cur_res += ',\n' + genSingleCase(dtype = dtype, params_list=param)

        if i % 5 == 0:
            count += 1
            B = np.random.randint(1,2)
            C = np.random.randint(1023,4096)
            M = np.random.randint(1023,4096)
            N = np.random.randint(1023,4096)
            param = [B, C, M, N, nan_inf]
            cur_res += ',\n' + genSingleCase(dtype = dtype, params_list=param)
    cur_res += '\n     ]\n}'
    print("the count of cases:", count)
    return cur_res

if __name__ == "__main__":
    res = '{\n\
    "op_name":"three_interpolate_forward",\n\
    "device":"gpu",\n\
    "require_value":true,\n\
    "evaluation_criterion":["diff1","diff2"],\n\
    "evaluation_threshold":[0.003,0.003],\n'
    dtype = "float32"
    res_fp32 = res + genCase(dtype)
    file = open("./three_interpolate_forward_random_float32.json",'w')
    file.write(res_fp32)
    res_fp32_nan_inf = res + genCase(dtype, True)
    file = open("./three_interpolate_forward_random_float32_nan_and_inf.json",'w')
    file.write(res_fp32_nan_inf)
    dtype = "float16"
    res_fp16 = res + genCase(dtype)
    file = open("./three_interpolate_forward_random_float16.json",'w')
    file.write(res_fp16)
    res_fp16_nan_inf = res + genCase(dtype, True)
    file = open("./three_interpolate_forward_random_float16_nan_and_inf.json",'w')
    file.write(res_fp16_nan_inf)
    file.close()
