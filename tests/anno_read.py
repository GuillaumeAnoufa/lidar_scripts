import numpy as np
import math

class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.cls_type = label[0]
        self.corners = np.array(label[1:], dtype=np.float32).reshape([8, 3])
        self.cornersToBox()
    
    def cornersToBox(self):
        box = self.corners
        angle = math.atan2(box[1, 1] - box[2, 1], box[1, 0] - box[2, 0])
        dim = np.max(box, axis=0) - np.min(box, axis=0) # max sur les colonnes
        center = (box[0] + box[6]) /2
        self.gt_box = np.concatenate([center, dim, angle], axis=None)

def get_objects_from_label(file):
    with open(file) as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects
        