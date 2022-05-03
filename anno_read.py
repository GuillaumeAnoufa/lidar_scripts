import numpy as np

def get_label(self, idx):
    label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
    assert label_file.exists()
    return self.get_objects_from_label(label_file)

def get_objects_from_label(file):
    with open(file) as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects
    
    
class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.class_ = label[0]
        self.corners = np.array(label[1:], dtype=np.float32)