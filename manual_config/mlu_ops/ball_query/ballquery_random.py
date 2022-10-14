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
def genSingleCase(dtype='float32', params_list=[1,2,1,0,0.1,1,0,1]):
    B = params_list[0]
    N = params_list[1]
    M = params_list[2]
    min_radius = params_list[3]
    max_radius = params_list[4]
    nsample = params_list[5]
    up_limit= params_list[6]
    bottom_limit=params_list[7]
  

    new_xyz_shape = [B, M, 3]
    xyz_shape     = [B, N, 3]
    output1_shape = [B, M, nsample]
    
    inputs = '     {\n       "inputs":['
    input1 = '{' + dShape(new_xyz_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(bottom_limit,up_limit) + "," + dlayout("ARRAY") + '}'
    input2 = '{' + dShape(xyz_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(bottom_limit,up_limit) + "," + dlayout("ARRAY") + '}'
    
    outputs = '       "outputs":['
    output1 = '{' + dShape(output1_shape) + ',' + dType('int32') + ',' + dlayout("ARRAY") + '}'


    inputs += input1 + ',\n' + '                 ' + input2 + '],\n'
    outputs += output1  + '],\n'
    
    min_radius_param = '"min_radius": ' + str(min_radius)
    max_radius_param = '"max_radius": ' + str(max_radius)
    nsample_param    = '"nsample": ' + str(nsample)
    
    op_params = '       "op_params":{' + min_radius_param + ',' + max_radius_param + ',' + nsample_param + '},\n'
    
    proto_param = '       "proto_params":{"write_data":true}'
    
    cur_res = inputs + outputs + op_params + proto_param + '\n     }'
    return cur_res

def genCase(dtype = "float32"):
    count = 1
    cur_res = '     "manual_data":[\n'
    B = np.random.randint(1,25)
    N = np.random.randint(1,512)
    M = np.random.randint(1,N + 1)
    max_radius = np.random.uniform(0, 2)
    min_radius = np.random.uniform(0, max_radius)
    nsample = np.random.randint(1,N + 1)
    up_limit = 1
    bottom_limit = -1
    param = [B, N, M, min_radius, max_radius, nsample, up_limit, bottom_limit]
    
    cur_res += genSingleCase(dtype = dtype, params_list=param)
    
    for i in range(15):
        count += 1
        B = np.random.randint(1, 25)
        N = np.random.randint(1, 1024)
        M = np.random.randint(1,N + 1)
        max_radius = np.random.uniform(0, 2)
        min_radius = np.random.uniform(0, max_radius)
        nsample = np.random.randint(1,N + 1)
        up_limit = 1
        bottom_limit = -1
        param = [B, N, M, min_radius, max_radius, nsample, up_limit, bottom_limit]

        cur_res += ',\n' + genSingleCase(dtype = dtype, params_list=param)
        if i % 2 == 0:
            count += 1
            B = np.random.randint(25, 50)
            N = np.random.randint(1024, 2048)
            M = np.random.randint(1,N + 1)
            max_radius = np.random.uniform(0, 2)
            min_radius = np.random.uniform(0, max_radius)
            nsample = np.random.randint(1,N + 1)
            up_limit = 1
            bottom_limit = -1
            param = [B, N, M, min_radius, max_radius, nsample, up_limit, bottom_limit]
            cur_res += ',\n' + genSingleCase(dtype = dtype, params_list=param)

        if i % 3 == 0:
            count += 1
            B = np.random.randint(50, 75)
            N = np.random.randint(2048, 3072)
            M = np.random.randint(1,N + 1)
            max_radius = np.random.uniform(0, 2)
            min_radius = np.random.uniform(0, max_radius)
            nsample = np.random.randint(1,N + 1)
            up_limit = 1
            bottom_limit = -1
            param = [B, N, M, min_radius, max_radius, nsample, up_limit, bottom_limit]
            cur_res += ',\n' + genSingleCase(dtype = dtype, params_list=param)
            
        if i % 5 == 0:
            count += 1
            B = np.random.randint(75, 100)
            N = np.random.randint(3072, 4096)
            M = np.random.randint(1,N + 1)
            max_radius = np.random.uniform(0, 2)
            min_radius = np.random.uniform(0, max_radius)
            nsample = np.random.randint(1,N + 1)
            up_limit = 1
            bottom_limit = -1
            param = [B, N, M, min_radius, max_radius, nsample, up_limit, bottom_limit]
            cur_res += ',\n' + genSingleCase(dtype = dtype, params_list=param)
    cur_res += '\n     ]\n}'
    print("the count of cases:", count)
    return cur_res

if __name__ == "__main__":
    res = '{\n\
    "op_name":"ball_query",\n\
    "device":"gpu",\n\
    "supported_mlu_platform":["370"],\n\
    "data_type":{"input_dtype":["float32", "float32"], "output_dtype":["int32"]},\n\
    "require_value":true,\n\
    "evaluation_criterion":["diff1","diff2"],\n\
    "evaluation_threshold":[0, 0],\n'
    dtype = "float32"
    res += genCase(dtype)
    file = open("./ball_query_random_float.json",'w')
    file.write(res)
    file.close()