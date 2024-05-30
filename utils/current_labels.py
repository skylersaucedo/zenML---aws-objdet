### use this to store current labels

def one_hot_json():
    """convert labels into JSON packet"""
    
    categories = [
        {'class_id' : 0, 'name' : 'blue_tape'},
        {'class_id' : 1, 'name' : 'black_tape'},
        {'class_id' : 2, 'name' : 'gum'},
        {'class_id' : 3, 'name' : 'leaf'}
        ]
    
    return categories


def one_hot_label(label):
    """
    soon to be depreciated...
    one hot encode label from string to int
    """
    r = 4
    if label == 'blue_tape':
        r = 0
    if label == 'black_tape':
        r = 1
    if label == 'gum':
        r = 2
    if label == 'leaf':
        r = 3
        
    return r