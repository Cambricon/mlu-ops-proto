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
def dparam(op_param,value):
    if str(value) == "False":
        value = "false"
    if str(value) == "True":
        value = "true"
    return '"'+ op_param + '"' + ":" + str(value)

def genSingleCase(dtype='float32', params_list=[1,1,4,1,1,1,1,1,1,1,1,1,0.5,1]):
    
    min_sizes_shape = [params_list[0]]
    aspect_ratios_shape = [params_list[1]]
    variance_shape = [params_list[2]]
    max_sizes_shape = [params_list[3]]
    h = params_list[4]
    w = params_list[5]
    im_height = params_list[6]
    im_width = params_list[7]
    clip_value = params_list[8]
    flip_value = params_list[9]
    step_h = params_list[10]
    step_w = params_list[11]
    offset = params_list[12]
    min_max_aspect_ratios_order = params_list[13]

    inputs = '    {"inputs":['
    input1 = '{' + dShape(min_sizes_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(0,150) + ','+ dlayout("ARRAY") + '}'
    input2 = '               {' + dShape(aspect_ratios_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(0.1,10) + ',' + dlayout("ARRAY") + '}'
    input3 = '               {' + dShape(variance_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(0.1,1) + ',' + dlayout("ARRAY") + '}'
    input4 = '               {' + dShape(max_sizes_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(151,300) + ',' + dlayout("ARRAY") + '}'
    
    num_priors = min_sizes_shape[0] + min_sizes_shape[0] * aspect_ratios_shape[0]
    if flip_value:
        num_priors += min_sizes_shape[0] * aspect_ratios_shape[0]
    if max_sizes_shape!=None:
        num_priors += max_sizes_shape[0]

    output_shape=[h,w,num_priors,4]
    var_shape=[h,w,num_priors,4]
    outputs = '    "outputs":['
    output1 = '{' + dShape(output_shape) + ',' + dType(dtype) + ',' + dlayout("ARRAY") + '}'
    output2 = '               {' + dShape(output_shape) + ',' + dType(dtype) + ',' + dlayout("ARRAY") + '}'
    inputs += input1 + ',\n' + input2 + ',\n' + input3 + ',\n' + input4 + '],\n'
    outputs += output1 + ',\n' + output2 +']\n'
    
    op_params = '    "op_params":{' + dparam("height",h)+ ','+ dparam("width",w)+ ',' + dparam("im_height",im_height)+ ',' + dparam("im_width",im_width) + ','+ \
    dparam("flip",bool(int(flip_value)))+ ',' + dparam("clip",bool(int(clip_value))) + ','+ dparam("step_h",step_h) + ','+ dparam("step_w",step_w) + ','+ \
    dparam("offset",offset) + ','+  dparam("min_max_aspect_ratios_order",bool(int(min_max_aspect_ratios_order))) + '},\n'

    cur_res = inputs + op_params + outputs + '}'
    return cur_res

def genCase():
    cur_res = '     "manual_data":[\n'
    min_sizes = np.random.randint(1,10)
    aspect_ratios_sizes = np.random.randint(1,10)
    variance_sizes = 4
    max_sizes = min_sizes
    h = np.random.randint(1,300)
    w = np.random.randint(1,300)
    im_height = np.random.randint(1,300)
    im_width = np.random.randint(1,300)
    clip_value = np.random.randint(0,2)
    flip_value = np.random.randint(0,2)
    step_h = np.random.uniform(0.1,300)
    step_w = np.random.uniform(0.1,300)
    offset = np.random.uniform(0.1,1)
    min_max_aspect_ratios_value = np.random.randint(0,2)
    param = [min_sizes,aspect_ratios_sizes,variance_sizes,max_sizes,h,w,
    im_height,im_width,clip_value,flip_value,step_h,step_w,offset,min_max_aspect_ratios_value]

    cur_res += genSingleCase(params_list=param)
    for i in range(200):
        if i<50:
            min_sizes = np.random.randint(25,30)
            aspect_ratios_sizes = np.random.randint(1,30)
            variance_sizes = 4
            max_sizes = min_sizes
            h = np.random.randint(1,50)
            w = np.random.randint(1,50)
            im_height = np.random.randint(1,300)
            im_width = np.random.randint(1,300)
            clip_value = np.random.randint(0,2)
            flip_value = np.random.randint(0,2)
            step_h = np.random.uniform(0.1,6)
            step_w = np.random.uniform(0.1,6)
            offset = np.random.uniform(-100,100)
            min_max_aspect_ratios_value = np.random.randint(0,2)
            param = [min_sizes,aspect_ratios_sizes,variance_sizes,max_sizes,h,w,
            im_height,im_width,clip_value,flip_value,step_h,step_w,offset,min_max_aspect_ratios_value]
            cur_res += ',\n' + genSingleCase(params_list=param)
        elif 50<i<100:
            min_sizes = np.random.randint(15,25)
            aspect_ratios_sizes = np.random.randint(1,15)
            variance_sizes = 4
            max_sizes = min_sizes
            h = np.random.randint(50,100)
            w = np.random.randint(50,100)
            im_height = np.random.randint(1,300)
            im_width = np.random.randint(1,300)
            clip_value = np.random.randint(0,2)
            flip_value = np.random.randint(0,2)
            step_h = np.random.uniform(0.1,3)
            step_w = np.random.uniform(0.1,3)
            offset = np.random.uniform(-100,100)
            min_max_aspect_ratios_value = np.random.randint(0,2)
            param = [min_sizes,aspect_ratios_sizes,variance_sizes,max_sizes,h,w,
            im_height,im_width,clip_value,flip_value,step_h,step_w,offset,min_max_aspect_ratios_value]
            cur_res += ',\n' + genSingleCase(params_list=param)
        elif 100<i<150:
            min_sizes = np.random.randint(10,15)
            aspect_ratios_sizes = np.random.randint(15,20)
            variance_sizes = 4
            max_sizes = min_sizes
            h = np.random.randint(100,200)
            w = np.random.randint(100,200)
            im_height = np.random.randint(1,300)
            im_width = np.random.randint(1,300)
            clip_value = np.random.randint(0,2)
            flip_value = np.random.randint(0,2)
            step_h = np.random.uniform(0.1,1.5)
            step_w = np.random.uniform(0.1,1.5)
            offset = np.random.uniform(0.1,1)
            min_max_aspect_ratios_value = np.random.randint(0,2)
            param = [min_sizes,aspect_ratios_sizes,variance_sizes,max_sizes,h,w,
            im_height,im_width,clip_value,flip_value,step_h,step_w,offset,min_max_aspect_ratios_value]
            cur_res += ',\n' + genSingleCase(params_list=param)
        else:
            min_sizes = np.random.randint(1,10)
            aspect_ratios_sizes = np.random.randint(1,15)
            variance_sizes = 4
            max_sizes = min_sizes
            h = np.random.randint(200,300)
            w = np.random.randint(200,300)
            im_height = np.random.randint(1,300)
            im_width = np.random.randint(1,300)
            clip_value = np.random.randint(0,2)
            flip_value = np.random.randint(0,2)
            step_h = np.random.uniform(0.1,1)
            step_w = np.random.uniform(0.1,1)
            offset = np.random.uniform(0.1,1)
            min_max_aspect_ratios_value = np.random.randint(0,2)
            param = [min_sizes,aspect_ratios_sizes,variance_sizes,max_sizes,h,w,
            im_height,im_width,clip_value,flip_value,step_h,step_w,offset,min_max_aspect_ratios_value]
            cur_res += ',\n' + genSingleCase(params_list=param)
    cur_res += '\n      ]\n}'
    return cur_res

if __name__ == "__main__":
    res = '{\n\
    "op_name":"prior_box",\n\
    "device":"gpu",\n\
    "evaluation_criterion":["diff1", "diff2","diff3"],\n\
    "data_type":{"input_dtype":["float32","float32","float32","float32"], "output_dtype":["float32","float32"]},\n\
    "if_dynamic_threshold":"true",\n\
    "require_value":true,\n\
    '
    res += genCase()
    file = open("./prior_box_manual_random_float32.json",'w')
    file.write(res)
    file.close()
    