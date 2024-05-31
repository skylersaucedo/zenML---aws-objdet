### use this to store current labels
import numpy as np

def one_hot_json():
    """make list of items of interest"""

    categories = [
        {'class_id' : 0, 'name' : 'blue_tape'},
        {'class_id' : 1, 'name' : 'black_tape'},
        {'class_id' : 2, 'name' : 'gum'},
        {'class_id' : 3, 'name' : 'leaf'}
        ]
    
    return categories

def search(name, x):
    
    if len([e for e in x if e['name'] == name]) > 0:
        return [e for e in x if e['name'] == name][0]['class_id']  
    else:
        return np.NaN 

def one_hot_label(label):
    """return class_id from label name"""
    cats = one_hot_json()
    return search(label,cats)