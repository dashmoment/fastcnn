import os
import matplotlib.image as mpimg
import vanilla_ssd as van
import pickle
import cv2



data_path = '/home/dashmoment/dataset/demo/img'
output_path = '/home/dashmoment/dataset/demo/label'
#data_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOC_train'
#output_path = '/media/ubuntu/65db2e03-ffde-4f3d-8f33-55d73836211a/dataset/VOCdevkit/VOC_train/label'


batch_size = 1

ckpt_filename = '/home/dashmoment/dataset/model/ssd_300/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
v = van.vanilla_ssd_net('/gpu:0', ckpt_filename)

image_list = os.listdir(data_path)

fcount = 0

for fid in range(0,len(image_list)):
    
    fcount = fcount + 1
    
    print("Process:{}/{}".format(fid, len(image_list)))
    
    img_path = os.path.join(data_path, image_list[fid])
    out_file =  os.path.join(output_path, image_list[fid]+'.pickle')
    img = cv2.imread(img_path)
    
    glabel, glocation, gscore = v.inference(img)
    fglabel, fglocation, fgscore = v.sess.run(v.flatten_output(glabel, glocation, gscore))
    
    
    with open(out_file, 'wb') as f:
        pickle.dump([fglabel, fglocation, fgscore], f, protocol=pickle.HIGHEST_PROTOCOL)
