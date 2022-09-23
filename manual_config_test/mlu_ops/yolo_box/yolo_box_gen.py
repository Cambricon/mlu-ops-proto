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
def dBool(v):
    if v:
        return 'true'
    else:
        return 'false'
def dRandomDistribution(start, end):
    return '"random_distribution":{"uniform":[' + str(start) + ',' + str(end) + ']}'
def dlayout(data_layout):
    return '"layout":"' + data_layout + '"'
def genSingleCase(x_shape, img_size, anchor_shape, boxes_shape, scores_shape, params):
    inputs = '     {\n       "inputs":['
    input1 = '{' + dShape(x_shape) + ',' + dType("float32") + ',' + dRandomDistribution(-1,1) + "," + dlayout("ARRAY") + '}'
    input2 = '{' + dShape(img_size) + ',' + dType("int32") + ',' + dRandomDistribution(10,1000) + "," + dlayout("ARRAY") + '}'
    input3 = '{' + dShape(anchor_shape) + ',' + dType("int32") + ',' + dRandomDistribution(1,15) + "," + dlayout("ARRAY") + '}'
    
    outputs = '       "outputs":['
    output1 = '{' + dShape(boxes_shape) + ',' + dType('float32') + ',' + dlayout("ARRAY") + '}'
    output2 = '{' + dShape(scores_shape) + ',' + dType('float32') + ',' + dlayout("ARRAY") + '}'


    inputs += input1 + ',\n                 ' + input2 + ',\n                 ' + input3 + '],\n'
    outputs += output1 + ',\n                 ' +  output2 + '],\n'
    
    proto_param = '       "op_params":{"class_num":' + str(params[0]) + ',"conf_thresh":' + str(params[1]) + ',"downsample_ratio":' + str(params[2]) \
    + ',"clip_bbox":'+dBool(params[3]) + ',"scale_x_y":' + str(params[4]) + ',"iou_aware":'+ dBool(params[5]) + ',"iou_aware_factor":' + str(params[6]) + '}'
    cur_res = inputs + outputs + proto_param + '\n     }'
    return cur_res

def genCase():
    cur_res = '     "manual_data":[\n'

    for i in range(30):
        class_num = np.random.randint(1,100)
        conf_thresh = np.random.randint(1,100)/100.0
        ratios = [8, 16, 32]
        downsample_ratio = np.random.choice(ratios, p = [0.4, 0.3, 0.3])
        clip_bbox = np.random.randint(0, 2)
        scales = [1.0, 1.5, 2.0]
        scale_x_y = np.random.choice(scales, p = [0.7, 0.15, 0.15])
        iou_aware = np.random.randint(0, 2)
        awares = [0.5, 0.6, 1.0]
        iou_aware_factor = np.random.choice(awares, p = [0.7, 0.15, 0.15])
        print(type(iou_aware_factor))
        param = [class_num, conf_thresh, downsample_ratio, clip_bbox, scale_x_y, iou_aware, iou_aware_factor]

        x_n = np.random.randint(1,32)
        anchor = np.random.randint(1,12)
        x_c = anchor * (6 + class_num) if iou_aware else anchor * (5 + class_num)
        x_h = 0
        x_w = 0
        if np.random.randint(0, 2):
            if x_c * x_n < 50:
                x_h = np.random.randint(200, 500)
                x_w = np.random.randint(200, 500)
            else:
                x_h = np.random.randint(1, 33)
                x_w = np.random.randint(1, 33)
        else:
            if x_c * x_n < 50:
                x_h = np.random.randint(200, 500)
            elif x_c * x_n < 100:
                x_h = np.random.randint(100, 200)
            else:
                x_h = np.random.randint(1, 100)
            x_w = x_h
        x_shape = [x_n, x_c, x_h, x_w]
        img_size = [x_n, 2]
        anchor_shape = [anchor*2]
        box_shape = [x_n, anchor, 4, x_h * x_w]
        scores_shape = [x_n, anchor, class_num, x_h * x_w]
        print(box_shape)
        cur_res += genSingleCase(x_shape, img_size, anchor_shape, box_shape, scores_shape, param) + ',\n'

    cur_res += '\n     ]\n}'
    return cur_res

if __name__ == "__main__":
    res = '{\n\
    "op_name":"yolo_box",\n\
    "device":"gpu",\n\
    "require_value":true,\n\
    "evaluation_criterion":["diff1", "diff2"],\n\
    "evaluation_threshold":[3e-3, 3e-3],\n'
    res += genCase()
    file = open("./yolo_box_manual_0.json",'w')
    file.write(res)
    file.close()