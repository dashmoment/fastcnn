# Yolo tiny

## Vanilla baseline
### Test data: VOC 2007 test set


### iou > 0.5

Accuracy: 0.322449

Elapse Time: 0.029400


### iou > 0.4

Accuracy: 0.393362

Elapse time: 0.0254 sec


### iou > 0.3

Accuracy: 0.440705

Elapse time: 0.02773 sec

### iou > 0.2

Accuracy: 0.469017

Elapse time: 0.029657 sec


## model 1

init: Yes

loss fuction: l1norm + kldivergence

shrink ratio: 0.8

Epoch: 19


### iou > 0.5

Accuracy: 0.221688

Elapse time: 0.00896134


### iou > 0.2

Accuracy: 0.317775

Elapse time: 0.00862465


## model 2

init: yes

loss function: yolo loss

shrink ratio: 0.8

Epoch: 24

### iou > 0.5

Accuracy: 0.2713675

Elapse Time: 0.009809


### iou > 0.4

Accuracy: 0.35122863

Elapse Time: 0.008594


### iou > 0.3

Accuracy: 0.405582

Elapse time: 0.0092743 sec


### iou > 0.2

Accuracy: 0.436231

Elapse time: 0.009870 sec



# SSD

## Vanilla Baseline 
## Test VOC 2007 test

### iou > 0.5

Accuracy: 0.596420

Elapse tim:0.0257256 sec
