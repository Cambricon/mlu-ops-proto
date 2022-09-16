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
def genSingleCase(dtype='float32', params_list=[1,1,1,1,1,1,1,1,1]):
    N = params_list[0]
    A = params_list[1]
    H = params_list[2]
    W = params_list[3]

    pre_nms_top_n = int(params_list[4])
    post_nms_top_n = int(params_list[5])
    nms_thresh = params_list[6]
    min_size = params_list[7]
    eta = params_list[8]
    pixel_offset = params_list[9]

    scores_shape = [N, H, W, A]
    deltas_shape = [N, H, W, A * 4]
    anchors_shape = [H, W, A, 4]
    variances_shape = [H, W, A, 4]
    img_shape = [N, 2]

    rois_shape = [N*post_nms_top_n, 4]
    rpn_roi_probs_shape = [N*post_nms_top_n, 1]
    rpn_rois_num_shape = [N]
    rpn_rois_batch_size_shape = [1]

    bottom_limit = 10
    up_limit = 100
    mid = bottom_limit + 0.5 * (up_limit-bottom_limit)

    inputs = '    {\n       "inputs":[\n'
    scores_input = '            {' + dShape(scores_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(bottom_limit,up_limit) + "," + dlayout("ARRAY") + '},\n'
    deltas_input = '            {' + dShape(deltas_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(bottom_limit,up_limit) + "," + dlayout("ARRAY") + '},\n'
    anchors_input = '            {' + dShape(anchors_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(bottom_limit,up_limit) + "," + dlayout("ARRAY") + '},\n'
    variances_input = '            {' + dShape(variances_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(bottom_limit,up_limit) + "," + dlayout("ARRAY") + '},\n'
    img_input = '            {' + dShape(img_shape) + ',' + dType(dtype) + ',' + dRandomDistribution(mid,up_limit) + "," + dlayout("ARRAY") + '}\n'
    

    outputs = '       "outputs":[\n'
    rois_output = '            {' + dShape(rois_shape) + ',' + dType(dtype) + ',' + dlayout("ARRAY") + '},\n'
    rpn_roi_probs_output = '            {' + dShape(rpn_roi_probs_shape) + ',' + dType(dtype) + ',' + dlayout("ARRAY") + '},\n'
    
    rrpn_rois_num_output = '            {' + dShape(rpn_rois_num_shape) + ',' + dType('int32') + ',' + dlayout("ARRAY") + '},\n'
    rpn_rois_batch_size_output = '            {' + dShape(rpn_rois_batch_size_shape) + ',' + dType('int32') + ',' + dlayout("ARRAY") + '}\n'

    inputs = inputs + scores_input + deltas_input + anchors_input + variances_input +img_input + '            ],\n'
    outputs = outputs + rois_output + rpn_roi_probs_output + rrpn_rois_num_output + rpn_rois_batch_size_output + '            ],\n'
    
    proto_param = '       "op_params":{\n'
    pre_nms_top_n_op =  '            "pre_nms_top_n":' + str(pre_nms_top_n) + ',\n'
    post_nms_top_n_op =  '            "post_nms_top_n":' + str(post_nms_top_n) + ',\n'
    nms_thresh_op =  '            "nms_thresh":' + str(nms_thresh) + ',\n'
    min_size_op =  '            "min_size":' + str(min_size) + ',\n'
    eta_op =  '            "eta":' + str(eta) + ',\n'
    pixel_offset_op =  '            "pixel_offset": true \n'
    if(pixel_offset == 0):
        pixel_offset_op =  '            "pixel_offset": false \n'

    proto_param = proto_param + pre_nms_top_n_op + post_nms_top_n_op + nms_thresh_op + min_size_op + eta_op +  pixel_offset_op + '        }\n'

    cur_res = inputs + outputs + proto_param + '    }'
    return cur_res

def genCase():
    cur_res = '    "manual_data":[\n'

    random = 1
    if random == 1:
        N = np.random.randint(1,20)
        A = np.random.randint(1,50)
        H = np.random.randint(1,50)
        W = np.random.randint(1,50)

        dn = min(A*H*W/3, 2000)
        up = max(A*H*W, 2000)
        pre_nms_top_n = up
        post_nms_top_n = dn

        nms_thresh = np.random.randint(1,100)/100
        min_size = np.random.randint(1,50)/100
        eta = 1.0
        pixel_offset = np.random.randint(0,10) > 5

        param = [N, A, H ,W, post_nms_top_n, pre_nms_top_n, nms_thresh, min_size, eta, pixel_offset]
        cur_res += genSingleCase(params_list=param)

        for i in range(60):
            N = np.random.randint(1,100)
            H = np.random.randint(1,100)
            W = np.random.randint(1,10)
            A = np.random.randint(1,10)

            dn = min(A*H*W/3, 2000)
            up = max(A*H*W, 2000)
            pre_nms_top_n = int(up)
            post_nms_top_n = int(dn)

            nms_thresh = np.random.randint(1,100)/100
            min_size = np.random.randint(1,50)/100 
            eta = 1.0
            pixel_offset = np.random.randint(0,10) > 5

            param = [N, A, H ,W, post_nms_top_n, pre_nms_top_n, nms_thresh, min_size, eta, pixel_offset]
            cur_res += ',\n' + genSingleCase(params_list=param)

            if i % 2 == 0:
                N = np.random.randint(1,200)
                H = np.random.randint(1,20)
                W = np.random.randint(1,20)
                A = np.random.randint(1,20)

                dn = min(A*H*W/3, 2000)
                up = max(A*H*W, 2000)
                pre_nms_top_n = int(up)
                post_nms_top_n = int(dn)

                nms_thresh = np.random.randint(1,100)/100
                min_size = np.random.randint(1,50)/100
                eta = 1.0
                pixel_offset = np.random.randint(0,10) > 5

                param = [N, A, H ,W, post_nms_top_n, pre_nms_top_n, nms_thresh, min_size, eta, pixel_offset]
                cur_res += ',\n' + genSingleCase(params_list=param)


            if i % 3 == 0:
                N = np.random.randint(1,200)
                A = np.random.randint(1,50)
                H = np.random.randint(1,100)
                W = np.random.randint(1,50)

                post_nms_top_n = np.random.randint(1,20000)
                pre_nms_top_n = np.random.randint(1,2000)
                nms_thresh = np.random.randint(1,100)/100
                min_size = np.random.randint(1,50)/100
                eta = 1.0
                pixel_offset = np.random.randint(0,10) > 5
                param = [N, A, H ,W, post_nms_top_n, pre_nms_top_n, nms_thresh, min_size, eta, pixel_offset]
                cur_res += ',\n' + genSingleCase(params_list=param)

                
            if i % 5 == 0:
                N = np.random.randint(1,100)
                A = np.random.randint(1,200)
                H = np.random.randint(1,50)
                W = np.random.randint(1,50)

                post_nms_top_n = np.random.randint(1,20000)
                pre_nms_top_n = np.random.randint(1,2000)
                nms_thresh = np.random.randint(1,100)/100
                min_size = np.random.randint(1,50)/100
                eta = 1.0
                pixel_offset = np.random.randint(0,10) > 5
                param = [N, A, H ,W, post_nms_top_n, pre_nms_top_n, nms_thresh, min_size, eta, pixel_offset]
                cur_res += ',\n' + genSingleCase(params_list=param)

    else:
        post_nms_top_n = 12000
        pre_nms_top_n = 2000
        nms_thresh = 0.5
        min_size = 0.1
        eta = 1.0
        pixel_offset = 1

        scale1=[1, 15, 54, 40]
        param=[post_nms_top_n, pre_nms_top_n, nms_thresh, min_size, eta, pixel_offset]
        scales=[scale1+param]

        cur_res += genSingleCase(params_list=scales[0])

        scales.append([1, 15, 48, 48])
        scales.append([1, 15, 42, 56])
        scales.append([1, 15, 40, 61])
        scales.append([1, 15, 44, 59])
        scales.append([1, 15, 59, 44])
        scales.append([1, 15, 42, 64])
        scales.append([1, 15, 48, 64])
        scales.append([1, 3, 50, 68])
        scales.append([1, 3, 160, 216])
        scales.append([1, 3, 160, 248])
        scales.append([1, 3, 176, 240])
        scales.append([1, 3, 168, 256])
        scales.append([1, 3, 184, 248])
        scales.append([1, 3, 248, 184])
        scales.append([1, 3, 176, 264])
        scales.append([1, 3, 22, 220])
        scales.append([1, 3, 264, 192])
        scales.append([1, 3, 20, 27])
        scales.append([1, 3, 200, 272])
        scales.append([1, 3, 200, 304])
        scales.append([1, 3, 20, 31])
        scales.append([1, 3, 22, 30])
        scales.append([1, 3, 21, 32])
        scales.append([1, 3, 23, 31])
        scales.append([1, 3, 31, 23])
        scales.append([1, 3, 22, 33])
        scales.append([1, 3, 88, 88])
        scales.append([1, 3, 33, 24])
        scales.append([1, 3, 25, 34])
        scales.append([1, 3, 80, 108])
        scales.append([1, 3, 25, 38])
        scales.append([1, 3, 80, 124])
        scales.append([1, 3, 88, 120])
        scales.append([1, 3, 84, 128])
        scales.append([1, 3, 124, 92])
        scales.append([1, 3, 92, 124])
        scales.append([1, 3, 88, 132])
        scales.append([1, 3, 11, 11])
        scales.append([1, 3, 132, 96])
        scales.append([1, 3, 100, 136])
        scales.append([1, 3, 10, 14])
        scales.append([1, 3, 100, 152])
        scales.append([1, 3, 10, 16])
        scales.append([1, 3, 11, 15])
        scales.append([1, 3, 11, 16])
        scales.append([1, 3, 11, 17])
        scales.append([1, 3, 16, 12])
        scales.append([1, 3, 44, 44])
        scales.append([1, 3, 17, 12])
        scales.append([1, 3, 40, 54])
        scales.append([1, 3, 13, 17])
        scales.append([1, 3, 13, 19])
        scales.append([1, 3, 40, 62])
        scales.append([1, 3, 44, 60])
        scales.append([1, 3, 42, 64])
        scales.append([1, 3, 46, 62])
        scales.append([1, 3, 62, 46])
        scales.append([1, 3, 44, 66])
        scales.append([1, 3, 176, 176])
        scales.append([1, 3, 66, 48])

        for scale in scales:
            cur_res += ',\n' + genSingleCase(params_list=(scale+param))
      
    cur_res += '\n    ]\n}'
    return cur_res

if __name__ == "__main__":
    res = '{\n\
    "op_name":"generate_proposals_v2",\n\
    "device":"gpu",\n\
    "require_value":true,\n\
    "evaluation_criterion":["diff1","diff2","diff3"],\n\
    "if_dynamic_threshold": true,\n'
    res += genCase()
    file = open("./generate_proposals_v2_random.json",'w')
    file.write(res)
    file.close()
 