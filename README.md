###Yolo tiny

##Vanilla baseline:
Test data: VOC 2007 test set
iou > 0.5
Accuracy: 0.322449
Elapse Time: 0.029400

iou > 0.4
Accuracy: 0.393362
Elapse time: 0.0254 sec

iou > 0.3
Accuracy: 0.440705
Elapse time: 0.02773 sec


## model 1. 
init: Yes
loss fuction: l1norm + kldivergence
shrink ratio: 0.8
Epoch: 19
-----------------------------
Accuracy: 0.10223
Elapse time: 0.0096619

## model 2
init: yes
loss function: yolo loss
shrink ratio: 0.8
Epoch: 24
-----------------------------
iou > 0.5
Accuracy: 0.2713675
Elapse Time: 0.009809

iou > 0.4
Accuracy: 0.35122863
Elapse Time: 0.008594

iou > 0.3
Accuracy: 0.405582
Elapse time: 0.0092743 sec



