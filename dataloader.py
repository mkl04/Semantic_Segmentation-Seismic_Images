import numpy as np
import random
import cv2
import os
from os.path import join as pjoin
from keras.utils import to_categorical
from keras import backend as K

from utils import make_divisible


def split_train_val(im_shape=(401,701), loader_type='section', n_groups=10, per_val=0.3, loc='sd'):
    '''Create inline and crossline 2D sections for training and validation'''
    
    if loc=='sd':
        path_data = '/scratch/parceirosbr/maykol.trinidad/dataset/F3'
    else:
        path_data = 'data'

    iline, xline = im_shape
    
    vert_locations = range(0, iline-n_groups, n_groups)
    horz_locations = range(0, xline-n_groups, n_groups)
    
    p_tr_list = []
    p_vl_list = []
    for i in vert_locations:
        vert_slides = np.arange(i, i+n_groups)
        
        aux = list(vert_slides[:-int(per_val*n_groups)])
        p_tr_list += ['i_' + str(ii) for ii in aux if ii < iline]
        
        aux = list(vert_slides[-int(per_val*n_groups):])
        p_vl_list += ['i_' + str(ii) for ii in aux if ii < iline]
    
    random.shuffle(p_tr_list)
    random.shuffle(p_vl_list)
    
    aux1 = []
    aux2 = []
    for j in horz_locations:
        horz_slides = np.arange(j, j+n_groups)
        
        aux = list(horz_slides[:-int(per_val*n_groups)])
        aux1 += ['x_' + str(jj) for jj in aux if jj < xline]
        
        aux = list(horz_slides[-int(per_val*n_groups):])
        aux2 += ['x_' + str(jj) for jj in aux if jj < xline]
    random.shuffle(aux1)
    random.shuffle(aux2)
    
    p_tr_list+=aux1
    p_vl_list+=aux2
    
    path_splits = pjoin(path_data, 'new_splits')
    
    if not os.path.exists(path_splits):
        os.makedirs(path_splits)
    
    file_object = open(pjoin(path_splits, loader_type + '_train.txt'), 'w')
    file_object.write('\n'.join(p_tr_list))
    file_object.close()
    
    file_object = open(pjoin(path_splits, loader_type + '_val.txt'), 'w')
    file_object.write('\n'.join(p_vl_list))
    file_object.close()


class section_loader():
    ''' Section loader of seismic blocks'''

    def __init__(self, direct="i", split='train', loc='sd'):

        if loc=="sd":
            self.root = '/scratch/parceirosbr/maykol.trinidad/dataset/F3'
        else:
            self.root = 'data'
        self.split = split
        self.n_classes = 6 
        self.mean = 0.000941 # average of the training data  
        self.direct = direct

        # Normal train/val mode
        self.seismic = np.load(pjoin(self.root,'train','train_seismic.npy'))
        self.labels  = np.load(pjoin(self.root,'train','train_labels.npy' ))
        self.labels  = to_categorical(self.labels,num_classes=self.n_classes)

        #for split in ['train', 'val', 'train_val']:
        path = pjoin(self.root, 'splits', 'section_' + self.split + '.txt')
        patch_list = tuple(open(path, 'r'))
        self.patch_list = patch_list
        
        self.index_direct = []
        
        for indexes in self.patch_list:
            direction, number = indexes.split(sep='_')
            if direction == self.direct:
                self.index_direct.append(number)

    def __len__(self):
        return len(self.index_direct)
    
    def generator(self):
    
        for number in self.index_direct:

            if self.direct == 'i':
                im = self.seismic[int(number),:,:]
                lbl = self.labels[int(number),:,:,:]
            elif self.direct == 'x':    
                im = self.seismic[:,int(number),:]
                lbl = self.labels[:,int(number),:,:]
            
            if im.shape[0] == 701:
                im = cv2.resize(im,(256,688))
                lbl = np.round(cv2.resize(lbl,(256,688)))
                
            else:
                im = cv2.resize(im,(256,400))  
                lbl = np.round(cv2.resize(lbl,(256,400)))
            
            im = np.expand_dims(im, axis = -1)

            yield im, lbl


def F3_generator(dataset, _bs):
    ''' Special generator to train with more than one images'size '''
    
    datasets = []
    for data in dataset:
        datasets.append(data.batch(_bs)) # no shuffle
        
    _batch=[]
    for ds in datasets:
        _iter = ds.repeat().make_one_shot_iterator()
        _batch.append(_iter.get_next())
    
    while True:
        for batch in _batch:
            x_batch, y_batch = K.get_session().run(batch) 
            yield x_batch, y_batch


def section_loader_test(model, split='test1', get_prob=False, loc='sd'):
    """
    Section loader of seismic bloc for inference

    Parameters
    ----------
    model : Model
        trained model
    split : str
        test options: 'test1' or 'test2'
    get_prob : bool
        return probabilities or argmax values
    loc : str
        location of test data
    """

    if loc=="sd":
        root = '/scratch/parceirosbr/maykol.trinidad/dataset/F3'
    else:
        root = 'data'

    seismic = np.load(pjoin(root,'test_once', split + '_seismic.npy'))
    
    #for split in ['test1', 'test2']:
    path = pjoin(root, 'splits', 'section_' + split + '.txt')
    patch_list = tuple(open(path, 'r'))
    
    output_p = np.zeros(seismic.shape + (6,))

    for indexes in patch_list:

        direction, number = indexes.split(sep='_')
        if direction == 'i':
            im  = seismic[int(number),:,:]
        elif direction == 'x':    
            im  = seismic[:,int(number),:]
        
        img_size = im.shape[:2]
        a, b = make_divisible(img_size)
        
        im = cv2.resize(im,(a,b))
        im = np.expand_dims(im, axis = -1)
        im = np.expand_dims(im, axis =  0)
        
        model_output = model.predict(im)
        model_output = cv2.resize(model_output[0],(img_size[1], img_size[0]))
            
        if direction == 'i':
            output_p[int(number),:,:,:] += model_output
            
        if direction == 'x':
            output_p[:,int(number),:,:] += model_output
            
    if get_prob:
        return output_p/2.
    return np.argmax(output_p,axis=-1)

##########################
# N to 1
##########################

class section_loader_ts():
    ''' Time-steps section loader of seismic blocks - N-to-1'''

    def __init__(self, direct="i", split='train', window=5, loc='sd'):
        
        if loc=="sd":
            self.root = '/scratch/parceirosbr/maykol.trinidad/dataset/F3'
        else:
            self.root = 'data'
        self.split = split
        self.n_classes = 6 
        self.direct = direct
        self.window = window

        # Normal train/val mode
        self.seismic = np.load(pjoin(self.root,'train','train_seismic.npy'))
        self.labels  = np.load(pjoin(self.root,'train','train_labels.npy' ))
        self.labels  = to_categorical(self.labels, num_classes=self.n_classes)

        #for split in ['train', 'val']:
        path = pjoin(self.root, 'splits', 'section_' + self.split + '.txt')
        patch_list = tuple(open(path, 'r'))
        self.patch_list = patch_list
        
        self.index_direct = []
        
        for indexes in self.patch_list:
            direction, number = indexes.split(sep='_')
            # ignore the first slides
            if direction == self.direct and int(number) not in np.arange(0,self.window-1):
                self.index_direct.append(number)

    def __len__(self):
        return len(self.index_direct)
    
    def generator(self):

        for number in self.index_direct:
            number = int(number)

            rand_val = np.random.uniform(0, 1)
            if rand_val > 0.5: # add both directions like data augmentation
                self.seismic = np.flip(self.seismic, axis=0)
                self.labels = np.flip(self.labels, axis=0)

            if self.direct == 'i':
                im = self.seismic[(number-self.window+1):(number+1),:,:]
                lbl = self.labels[number,:,:,:]
            elif self.direct == 'x':    
                im = self.seismic[:,(number-self.window+1):(number+1),:]
                lbl = self.labels[:,number,:,:]
            
            if im.shape[0] == 401:
                im = [np.expand_dims(cv2.resize(im[:,idx,:], (256,400)) , axis=0) for idx in range(self.window)]
                im = np.vstack(im)
                lbl = np.round(cv2.resize(lbl,(256,400)))

            else:
                im = [np.expand_dims(cv2.resize(im[idx,:,:], (256,688)) , axis=0) for idx in range(self.window)]
                im = np.vstack(im)
                lbl = np.round(cv2.resize(lbl,(256,688)))
            
            im = np.expand_dims(im, axis = -1)

            yield im, lbl


def section_loader_test_ts(model, split='test1', get_prob=False, loc='sd', window=5):

    if loc=="sd":
        root = '/scratch/parceirosbr/maykol.trinidad/dataset/F3'
    else:
        root = 'data'

    seismic = np.load(pjoin(root,'test_once', split + '_seismic.npy'))
    
    #for split in ['test1', 'test2']:
    path = pjoin(root, 'splits', 'section_' + split + '.txt')
    patch_list = tuple(open(path, 'r'))
    
    output_p = np.zeros(seismic.shape + (6,))

    for indexes in patch_list:

        direction, number = indexes.split(sep='_')
        number = int(number)
        
        if number in np.arange(0,window-1):
            idx_l = number
            idx_r = number+window
            flip_ = True
        else:
            idx_l = number-window+1
            idx_r = number+1
            flip_ = False
            
        if direction == 'i':
            im = seismic[idx_l:idx_r,:,:]
        elif direction == 'x':    
            im = seismic[:,idx_l:idx_r,:]
            im = np.rollaxis(im, 1, 0)
        
        if flip_: # for first slides, we have to change of direction of timesteps images
            im = np.flip(im, axis=0) 
        
        img_size = im.shape[1:3] # H, W
        a, b = make_divisible(img_size)
        
        im = np.vstack([np.expand_dims(cv2.resize(im[idx], (a,b)) , axis=0) for idx in range(window)])
        
        im = np.expand_dims(im, axis = -1)
        im = np.expand_dims(im, axis =  0)
        
        model_output = model.predict(im)
        model_output = cv2.resize(model_output[0], (img_size[1], img_size[0]))
            
        if direction == 'i':
            output_p[number,:,:,:] += model_output
            
        if direction == 'x':
            output_p[:,number,:,:] += model_output
            
    if get_prob:
        return output_p/2.
    
    return np.argmax(output_p,axis=-1)

##########################
# N to N
##########################

class section_loader_ts_n2n():
    ''' Time-steps section loader of seismic blocks - N-to-N'''

    def __init__(self, direct="i", split='train', window=5, loc='sd'):
        
        if loc=="sd":
            self.root = '/scratch/parceirosbr/maykol.trinidad/dataset/F3'
        else:
            self.root = 'data'
        self.split = split
        self.n_classes = 6 
        self.direct = direct
        self.window = window

        # Normal train/val mode
        self.seismic = np.load(pjoin(self.root,'train','train_seismic.npy'))
        self.labels  = np.load(pjoin(self.root,'train','train_labels.npy' ))
        self.labels  = to_categorical(self.labels, num_classes=self.n_classes)

        #for split in ['train', 'val']:
        path = pjoin(self.root, 'splits', 'section_' + self.split + '.txt')
        patch_list = tuple(open(path, 'r'))
        self.patch_list = patch_list
        
        self.index_direct = []
        
        for indexes in self.patch_list:
            direction, number = indexes.split(sep='_')
            # ignore the first slides
            if direction == self.direct and int(number) not in np.arange(0,self.window-1):
                self.index_direct.append(number)

    def __len__(self):
        return len(self.index_direct)
    
    def generator(self):
        
        for number in self.index_direct:
            number = int(number)
            
            rand_val = np.random.uniform(0, 1)
            if rand_val > 0.5: # add both directions like data augmentation
                self.seismic = np.flip(self.seismic, axis=0)
                self.labels = np.flip(self.labels, axis=0)
            
            if self.direct == 'i':
                im = self.seismic[(number-self.window+1):(number+1),:,:]
                lbl = self.labels[(number-self.window+1):(number+1),:,:] # changes
            elif self.direct == 'x':    
                im = self.seismic[:,(number-self.window+1):(number+1),:]
                lbl = self.labels[:,(number-self.window+1):(number+1),:] # changes
                
            if im.shape[0] == 401:
                im = [np.expand_dims(cv2.resize(im[:,idx,:], (256,400)) , axis=0) for idx in range(self.window)]
                im = np.vstack(im)
                # changes
                lbl = [np.expand_dims(cv2.resize(lbl[:,idx,:], (256,400)) , axis=0) for idx in range(self.window)]
                lbl = np.vstack(lbl)

            else:
                im = [np.expand_dims(cv2.resize(im[idx,:,:], (256,688)) , axis=0) for idx in range(self.window)]
                im = np.vstack(im)
                # changes
                lbl = [np.expand_dims(cv2.resize(lbl[idx,:,:], (256,688)) , axis=0) for idx in range(self.window)]
                lbl = np.vstack(lbl)
                
            im = np.expand_dims(im, axis = -1)

            yield im, lbl


def section_loader_test_ts_n2n(model, split='test1', get_prob=False, loc='sd', window=5):

    if loc=="sd":
        root = '/scratch/parceirosbr/maykol.trinidad/dataset/F3'
    else:
        root = 'data'

    seismic = np.load(pjoin(root,'test_once', split + '_seismic.npy'))
    
    #for split in ['test1', 'test2']:
    path = pjoin(root, 'splits', 'section_' + split + '.txt')
    patch_list = tuple(open(path, 'r'))
    
    output_p = np.zeros(seismic.shape + (6,))

    for indexes in patch_list:

        direction, number = indexes.split(sep='_')
        number = int(number)
        
        if number in np.arange(0,window-1):
            idx_l = number
            idx_r = number+window
            flip_ = True
        else:
            idx_l = number-window+1
            idx_r = number+1
            flip_ = False
            
        if direction == 'i':
            im = seismic[idx_l:idx_r,:,:]
        elif direction == 'x':    
            im = seismic[:,idx_l:idx_r,:]
            im = np.rollaxis(im, 1, 0)
        
        if flip_: # for first slides, we have to change of direction of timesteps images
            im = np.flip(im, axis=0) 
        
        img_size = im.shape[1:3] # H, W
        a, b = make_divisible(img_size)
        
        im = np.vstack([np.expand_dims(cv2.resize(im[idx], (a,b)) , axis=0) for idx in range(window)])
        
        im = np.expand_dims(im, axis = -1)
        im = np.expand_dims(im, axis =  0)

        model_output = model.predict(im)
        model_output = np.vstack([
            np.expand_dims(cv2.resize(model_output[0, idx], (img_size[1], img_size[0])) , axis=0) for idx in range(window)])
        
        if flip_: # for first slides, we have to return to the correct direction of timesteps images
            model_output = np.flip(model_output, axis=0) #changes
            
        if direction == 'i':
            #changes
            output_p[idx_l:idx_r,:,:,:] += model_output
            
        if direction == 'x':
            #changes
            model_output = np.rollaxis(model_output, 1, 0)
            output_p[:,idx_l:idx_r,:,:] += model_output
            
    if get_prob:
        return output_p/2.
    
    return np.argmax(output_p,axis=-1)