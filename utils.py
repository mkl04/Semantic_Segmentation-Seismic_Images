import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def zoom(x,y):

    def auxf(x,y):
        ratio = 0.75
        sh = tf.cast(tf.shape(x) , tf.float32)
        sh_lab = tf.cast(tf.shape(y) , tf.float32)
        crop_size_img = [tf.math.round(sh[0]*ratio), tf.math.round(sh[1]*ratio), sh[2]]
        crop_size_lab = [tf.math.round(sh[0]*ratio), tf.math.round(sh[1]*ratio), sh_lab[2]]
        x_crop = tf.random_crop(value = x, size = crop_size_img, seed = 42)
        y_crop = tf.random_crop(value = y, size = crop_size_lab, seed = 42)
        
        x_back = tf.image.resize_images(x_crop, size = tf.cast(sh[:2], tf.int32))
        y_back = tf.image.resize_images(y_crop, size = tf.cast(sh[:2], tf.int32))

        return tf.cast(x_back, tf.float64), tf.cast(tf.math.round(y_back), tf.float32)
    
    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply zoom 50% of the time
    return tf.cond(choice < 0.5, lambda: (x,y) , lambda: auxf(x,y))


def make_aug(dataset, type_aug):

    if type_aug == "aug1":
        dataset = dataset.map(zoom)

    # if type_aug == "aug2":
    #     dataset = dataset.concatenate(aug1).concatenate(aug2).concatenate(aug4)

    return dataset


def PlotHistory(_model, feature, start_epoch = 0, path_file = None):
    val = "val_" + feature
    
    plt.xlabel('Epoch Number - ' + str(start_epoch))
    plt.ylabel(feature)
    plt.plot(_model.history[feature][start_epoch:])
    plt.plot(_model.history[val][start_epoch:])
    plt.legend(["train_"+feature, val])    
    if path_file:
        plt.savefig(path_file)


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask].astype(int), minlength=n_class**2).reshape(n_class, n_class)
    return hist


def calculate_metrics_total(Y1, Y1p, Y2, Y2p):
    
    acc1 = (Y1==Y1p).sum()/np.prod(Y1.shape)
    print("Accuracy 1: ", np.round(acc1, 4))
    
    acc2 = (Y2==Y2p).sum()/np.prod(Y2.shape)
    print("Accuracy 2: ", np.round(acc2,4)) 
    
    # Metrics together:
    Yt  = np.concatenate((Y1.reshape(-1), Y2.reshape(-1)))
    Ypt = np.concatenate((Y1p.reshape(-1),Y2p.reshape(-1)))

    hist = _fast_hist(Yt,Ypt, 6)
    
    acc = np.diag(hist).sum() / hist.sum()
    print("Pixel Accuracy: ", np.round(acc,4))
    acc_cls = np.round(np.diag(hist) / hist.sum(axis=1), 3)[::-1]
    print("Class Accuracy: ", acc_cls)
    mean_acc_cls = np.nanmean(acc_cls)
    print("Mean Class Accuracy: ", mean_acc_cls)

    # Freq Weighted IoU:
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum() # fraction of the pixels that come from each class
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    print("FWIOU: ", np.round(fwavacc,4))
    print("mIoU: ", mean_iu)
    # print("freq:", freq)

def make_divisible(img_shape, div = 16, mode = 'sup'):
    a = (img_shape[1]//div)*div
    b = (img_shape[0]//div)*div
    if mode == 'sup':
        a += div
        b += div
    return a, b      