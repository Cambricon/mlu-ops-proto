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
    n = params_list[0]
    iou_threshold = params_list[1]
    up_limit= params_list[2]
    bottom_limit=params_list[3]

    output1_shape = [n]
    output2_shape = [1]
    input_shape = [n ,9]
    inputs = '     {\n       "inputs":['
    input1 = '{' + dShape(input_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(bottom_limit,up_limit) + "," + dlayout("ARRAY") + '}'
    
    outputs = '       "outputs":['
    output1 = '{' + dShape(output1_shape) + ',' + dType('int32') + ',' + dlayout("ARRAY") + '}'
    output2 = '{' + dShape(output2_shape) + ',' + dType('int32') + ',' + dlayout("ARRAY") + '}'


    inputs += input1 + '],\n'
    outputs += output1 + ',\n                 ' +  output2 + '],\n'
    
    proto_param = '       "op_params":{"iou_threshold":' + str(iou_threshold) + '}'
    cur_res = inputs + outputs + proto_param + '\n     }'
    return cur_res

def genCase():
    cur_res = '     "manual_data":[\n'
    n = np.random.randint(1,100)
    iou_threshold = np.random.randint(1,100)/100
    up_limit = np.random.randint(1,50)
    bottom_limit = np.random.randint(-50,0)
    param = [n, iou_threshold, up_limit,bottom_limit]

    cur_res += genSingleCase(params_list=param)
    for i in range(25):
        n = np.random.randint(100,500)
        iou_threshold = np.random.randint(1,100)/100
        up_limit = np.random.randint(1,500)
        bottom_limit = np.random.randint(-500,0)
        param = [n,iou_threshold,up_limit,bottom_limit]

        cur_res += ',\n' + genSingleCase(params_list=param)
        if i % 3 == 0:
            n = np.random.randint(500,1000)
            iou_threshold = np.random.randint(1,100)/100
            up_limit = np.random.randint(0,1000)
            bottom_limit = np.random.randint(-5000,0)
            param = [n,iou_threshold,up_limit,bottom_limit]
            cur_res += ',\n' + genSingleCase(params_list=param)
            
        if i%4 == 0:
            n = np.random.randint(1000,2000)
            iou_threshold = np.random.randint(1,100)/100
            up_limit = np.random.randint(1,2000)
            bottom_limit = np.random.randint(-2000, 0)

            param = [n,iou_threshold,up_limit,bottom_limit]
            cur_res += ',\n' + genSingleCase(params_list=param)
    cur_res += '\n     ]\n}'
    return cur_res

if __name__ == "__main__":
    res = '{\n\
    "op_name":"poly_nms",\n\
    "device":"gpu",\n\
    "require_value":true,\n\
    "evaluation_criterion":["diff3"],\n\
    "evaluation_threshold":[0],\n'
    res += genCase()
    file = open("./poly_nms_random.json",'w')
    file.write(res)
    file.close()