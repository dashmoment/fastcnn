import tensorflow as tf
import numpy as np
import cv2
import voc_utils as voc
import os
import utility as ut

def eval_by_img(imgname, testBB, iou_threshold):

    BB = []
    ann = voc.load_annotation(imgname)

    for item in ann.find_all('object'):
         
        xmin = float(item.xmin.contents[0])
        ymin = float(item.ymin.contents[0])
        xmax = float(item.xmax.contents[0])
        ymax = float(item.ymax.contents[0])
        
        tmp = [xmin, ymin, xmax, ymax]
        BB.append(tmp)

    match = matchBB(testBB, BB, iou_threshold)

    return match

def matchBB(testBB, gtBB, iou_threshold): #box = [xmin, ymin, xmax,ymax]
    
    match = []

    for gtbb in gtBB:

        if iou_new(testBB, gtbb) > iou_threshold:

            match.append(1)

        else:
            match.append(-1)

    return match


def vocimg_preprocess(fname):
    
    img = cv2.imread(fname)
    img_resized = cv2.resize(img, (448, 448))
    img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
    img_resized_np = np.asarray( img_RGB )
    
    res = np.zeros((1,448,448,3),dtype='float32')
    res[0] = (img_resized_np/255.0)*2.0-1.0

    return res

def iou_new(box1,box2): #box = [xmin, ymin, xmax,ymax]

    h1 = box1[3] - box1[1]
    h2 = box2[3] - box2[1]
    w1 = box1[2] - box1[0]
    w2 = box2[2] - box2[0]
    
    tb = min(box1[3],box2[3]) - max(box1[1],box2[1])
    lr = min(box1[2],box2[2]) - max(box1[0],box2[0])
    if tb < 0 or lr < 0 : intersection = 0
    else : intersection =  tb*lr
    return intersection / (h1*w1 + h2*w2 - intersection)

def iou(box1,box2): #box = [cx, cy, width,height]
        tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
        lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
        if tb < 0 or lr < 0 : intersection = 0
        else : intersection =  tb*lr
        return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

def interpret_output(intputs):
    
        threshold = 0.05
        iou_threshold = 0.2
        classes =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
        output =  np.copy(intputs)
    
        w_img = 448
        h_img = 448
        
        probs = np.zeros((7,7,2,20))
        class_probs = np.reshape(output[0:980],(7,7,20))
        scales = np.reshape(output[980:1078],(7,7,2))
        boxes = np.reshape(output[1078:],(7,7,2,4))
        offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))

        boxes[:,:,:,0] += offset
        boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
        boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
        boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
        boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])
        
        boxes[:,:,:,0] *= w_img
        boxes[:,:,:,1] *= h_img
        boxes[:,:,:,2] *= w_img
        boxes[:,:,:,3] *= h_img

        for i in range(2):
            for j in range(20):
                probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])

        filter_mat_probs = np.array(probs>=threshold,dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]] 

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]
       
        
        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0 : continue
            for j in range(i+1,len(boxes_filtered)):
                if iou(boxes_filtered[i],boxes_filtered[j]) > iou_threshold : 
                    probs_filtered[j] = 0.0
        
        filter_iou = np.array(probs_filtered>0.0,dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

        return result


def show_results(img,results):
        img_cp = img.copy()
       
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3])//2
            h = int(results[i][4])//2
            
            cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)
            cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
            
        cv2.imshow('YOLO_tiny detection',img_cp)
        cv2.waitKey(1000)
       