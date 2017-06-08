import os
import tensorflow as tf
import matplotlib.image as mpimg
import utility as ut
import vanilla_ssd as van
import pickle




data_path = '/home/dashmoment/dataset/demo/img'
output_path = '/home/dashmoment/dataset/demo/label'
batch_size = 1


v = van.vanilla_ssd_net()

image_list = os.listdir(data_path)

fcount = 0

for fid in range(len(image_list)):
    
    fcount = fcount + 1
    
    img_path = os.path.join(data_path, image_list[fid])
    out_file =  os.path.join(output_path, image_list[fid]+'.pickle')
    img = mpimg.imread(img_path)
    
    glabel, glocation, gscore = v.inference(img)
    fglabel, fglocation, fgscore = v.sess.run(v.flatten_output(glabel, glocation, gscore))
    
    
    with open(out_file, 'wb') as f:
        pickle.dump([fglabel, fglocation, fgscore], f, protocol=pickle.HIGHEST_PROTOCOL)
