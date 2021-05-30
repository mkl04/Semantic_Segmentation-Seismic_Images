import numpy as np
import os
from os.path import join as pjoin

import tensorflow as tf
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint

from dataloader import section_loader_ts, F3_generator
from models import BUnetConvLSTM_Nto1
from metrics import mIoU
from utils import make_aug, PlotHistory
from loss import *


# gpu
import argparse
import sys

parser = argparse.ArgumentParser(description='testing code')
parser.add_argument('--gpu_id', dest='gpu_id', type=str, default='0', help='set gpu id')
args = parser.parse_args()

# set gpu device id
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
print('selecting gpu  :', args.gpu_id)

temp = sys.stdout
sys.stdout = open('log001.txt', 'w') # redirect all prints to this log file 



class_weights = np.array([ 0.59325097,  1.40226066,  0.34299189,  2.5094202 ,  5.08341702, 11.04956761])

if args.gpu_id == "0":
    idd = 1
    lossf = 'categorical_crossentropy'
if args.gpu_id == "1":
    idd = 2
    lossf = weighted_categorical_crossentropy(class_weights)
if args.gpu_id == "2":
    idd = 3
    lossf = categorical_focal_loss(gamma=2,alpha=1)
if args.gpu_id == "3":
    idd = 5
    lossf = categorical_focal_loss(gamma=1,alpha=1)

bs = 1
_epochs = 100
verbose_train = 2
ts = 5 # windows
type_aug = "no_aug" #no_aug, aug1

model_name = "unet_bconvlstm"
f_conv = 16
f_convlstm = 64


# if args.gpu_id == "0":
#     idd = 5
#     lossf = categorical_focal_loss(gamma=1,alpha=1)
# if args.gpu_id == "1":
#     idd = 6
#     lossf = weighted_categorical_focal_loss(class_weights, gamma=1, alpha=1)
# if args.gpu_id == "2":
#     idd = 7
#     lossf = categorical_focal_loss(gamma=0.5,alpha=1)
# if args.gpu_id == "3":
#     idd = 8
#     lossf = weighted_categorical_focal_loss(class_weights, gamma=0.5, alpha=1)

# Load Data

train_loader_i = section_loader_ts(direct = 'i', split = 'train')
train_loader_x = section_loader_ts(direct = 'x', split = 'train')
val_loader_i = section_loader_ts(direct = 'i',split = 'val')
val_loader_x = section_loader_ts(direct = 'x',split = 'val')

dat_tr1 = tf.data.Dataset.from_generator(train_loader_i.generator, output_types = (tf.float64, tf.float32), 
                                         output_shapes = ((ts,688,256,1), (688,256,6)))
dat_tr2 = tf.data.Dataset.from_generator(train_loader_x.generator, output_types = (tf.float64, tf.float32),
                                        output_shapes = ((ts,400,256,1), (400,256,6)))

dat_vl1 = tf.data.Dataset.from_generator(val_loader_i.generator, output_types = (tf.float64, tf.float32),
                                        output_shapes = ((ts,688,256,1), (688,256,6)))
dat_vl2 = tf.data.Dataset.from_generator(val_loader_x.generator, output_types = (tf.float64, tf.float32),
                                        output_shapes = ((ts,400,256,1), (400,256,6)))


data_tr = [make_aug(dat_tr1, type_aug), make_aug(dat_tr2, type_aug)]
data_vl = [dat_vl1, dat_vl2]

num_train = len(train_loader_i) + len(train_loader_x)
num_val = len(val_loader_i) + len(val_loader_x)

train_gen = F3_generator(data_tr, bs)
val_gen = F3_generator(data_vl, bs)

# Modeling

model = BUnetConvLSTM_Nto1(6, f_conv, f_convlstm, ts)

# Training
path_res = '/scratch/parceirosbr/maykol.trinidad/DL_project/res'
name_weight = '{}_{}_ts{}_{}_f{}_fbn{}.h5'.format(model_name, idd, ts, type_aug, f_conv, f_convlstm)
filepath = pjoin(path_res, name_weight)
# filepath = '/scratch/parceirosbr/maykol.trinidad/DL_project/res/ResUnet_F3_' + str(idd) + '_' + type_aug + '_nb3_f3.h5' 


lr = 1e-4
stopPatience = 10
modelCheck = ModelCheckpoint(filepath, verbose=1, save_best_only=True, save_weights_only=True, period=1) # monitor="val_dice_loss", mode = 'min')
earlystopper = EarlyStopping(patience=stopPatience, verbose=1) # monitor="val_dice_loss", mode = 'min')
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=stopPatience//2, min_lr=0.0000005, verbose=1) # monitor="val_dice_loss", mode = 'min')
cb = [modelCheck, earlystopper, reduce_lr]

# opt = RMSprop(lr=lr)
opt = Adam(lr=lr, amsgrad=False)
model.compile(optimizer=opt,
              loss = lossf,
              metrics=['acc', mIoU])

model.fit_generator(train_gen,epochs=_epochs,
                    steps_per_epoch=num_train//bs,
                    callbacks=cb,
                    validation_data=val_gen, validation_steps=num_val//bs,
                    verbose = verbose_train, workers = 0)


# Testing best on training

print("==========================================================")
print(name_weight)

from dataloader import section_loader_test_ts
from utils import calculate_metrics_total

model.load_weights(filepath)

# path_data = '../dataset/F3'
path_data = '/scratch/parceirosbr/maykol.trinidad/dataset/F3'
labels1  = np.load(pjoin(path_data,'test_once', 'test1_labels.npy' ))
labels2  = np.load(pjoin(path_data,'test_once', 'test2_labels.npy' ))

labels_predict1 = section_loader_test_ts(model, 'test1')
labels_predict2 = section_loader_test_ts(model, 'test2')

calculate_metrics_total(labels1, labels_predict1, labels2, labels_predict2)

sys.stdout = temp
sys.stdout.close()