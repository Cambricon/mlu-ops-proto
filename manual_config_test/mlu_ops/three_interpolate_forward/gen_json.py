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
    input0 = '{' + dShape(input_shape_0) + ',' + dRandomDistribution(-10,10) + "," + dlayout("ARRAY") + '}'
    input1 = '{' + dShape(input_shape_1) + ',' + dRandomDistribution(0,m) + "," + dlayout("ARRAY") + '}'
    input2 = '{' + dShape(input_shape_2) + ',' + dRandomDistribution(-10,10) + "," + dlayout("ARRAY") + '}'
    
    output_shape_0 = [b, c, n]
    outputs = '       "outputs":['
    output0 = '{' + dShape(output_shape_0) + ',' + dlayout("ARRAY") + '}'


    inputs += input0 + ',\n                 ' +  input1 + ',\n                 ' +  input2 + '],\n'
    outputs += output0 + ']\n'
    
    cur_res = inputs + outputs + '\n     }'
    return cur_res

def genCase():
    cur_res = '     "manual_data":[\n'

    b = np.random.randint(1,1024)
    c = np.random.randint(10,100)
    m = np.random.randint(10,100)
    n = np.random.randint(10,100)
    param = [b, c, m, n]
    cur_res += genSingleCase(params_list=param)

    for i in range(15):
        if i % 4 == 0:
            b = np.random.randint(1,1024)
            c = np.random.randint(10,100)
            m = np.random.randint(10,100)
            n = np.random.randint(10,100)
            param = [b, c, m, n]
            cur_res += ',\n' + genSingleCase(params_list=param)

    for i in range(50):
        b = np.random.randint(1,64)
        c = np.random.randint(1,500)
        m = np.random.randint(1,200)
        n = np.random.randint(1,1000)
        param = [b, c, m, n]
        cur_res += ',\n' + genSingleCase(params_list=param)

    cur_res += '\n     ]\n}'
    return cur_res

if __name__ == "__main__":
    res = '{\n\
    "op_name":"three_interprolate_forward",\n\
    "device":"gpu",\n\
    "require_value":true,\n\
    "data_type":{"input_dtype":["float16", "int32", "float16"], "output_dtype":["float16"]},\n\
    "evaluation_criterion":["diff1", "diff2"],\n\
    "evaluation_threshold":[3e-3, 3e-3],\n'
    res += genCase()
    file = open("./manual_half_0.json",'w')
    file.write(res)
    file.close()