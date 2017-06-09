import os
import vanilla_ssd as van
import cv2
import voc_utils as voc
import time
import ssd_shrink_network as ssd_s

classes_label =  ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

def process_result(img, classes, scores, bboxes):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    
    height = img.shape[0]
    width = img.shape[1]
    
    result = []
  
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            result.append([xmin, ymin, xmax, ymax, classes_label[cls_id]])
           
    return  result   


def eval_by_obj(imgname, testBB, iou_threshold):

    BB = []
    ann = voc.load_annotation(imgname)

    for item in ann.find_all('object'):

        find_cls = item.find_all('name')
        class_name = find_cls[0].contents[0]

        xmin = float(item.xmin.contents[0])
        ymin = float(item.ymin.contents[0])
        xmax = float(item.xmax.contents[0])
        ymax = float(item.ymax.contents[0])
        
        tmp = [xmin, ymin, xmax, ymax,class_name]
        
        
        BB.append(tmp)


#    print(BB)
#    print(testBB)
    match = matchBB(testBB, BB, iou_threshold)

    return match

def matchBB(testBB, gtBB, iou_threshold): #box = [xmin, ymin, xmax,ymax]
    
    match = 0

    for gtbb in gtBB:

#        print("iou:{}".format(iou_new(testBB, gtbb)))

        if iou_new(testBB, gtbb) > iou_threshold and testBB[4] == gtbb[4]:

            match = 1
            break;
        else:
            match = -1
        
    return match
  
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


def calc_objec_num(imgname):

    nBB = 0

    ann = voc.load_annotation(imgname)
    
    for item in ann.find_all('object'):
        nBB = nBB + 1

    return nBB


img_root = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2007/Test/JPEGImages'
labelfiles = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC2007/Test/ImageSets/Main'

#v = van.vanilla_ssd_net('/gpu:1')
v = ssd_s.ssd_shrink_network('test', 1 , 1, '', '/gpu:1')


image_list = os.listdir(img_root)
val_list = voc.imgs_from_category_as_list('', 'test', labelfiles)

fcount = 0
tp = 0
fp = 0
num = 0
elapse = 0

#for fid in range(len(val_list)//4):
for fid in range(100):
    
    fcount = fcount + 1
    
    print("Process:{}/{}".format(fid, len(val_list)))
    
    
    img_path = os.path.join(img_root, val_list[fid]+'.jpg')

    img = cv2.imread(img_path)   
    img = v.img_preprocessing(img)
    
    
    s = time.clock()   
    label, score , location= v.inference(img)
    e = time.clock()
    
    print("time:{}".format(e-s))
    if fid != 0: elapse = elapse + e - s
    
    
#    label, score , location = v.predict_box(img) 
    
#    v.plot(img)
    
    num = num + calc_objec_num(val_list[fid])   
    res = process_result(img,  label, score , location)
    
    for j in range(len(res)):
        match  = eval_by_obj(val_list[fid],res[j],0.5)    
        if(match == 1): tp = tp +1
        if(match == -1): fp = fp +1
        

print("Avg Elapse:{}".format(elapse/99)) 
print("Accuracy:{}".format(tp/num))

        
        