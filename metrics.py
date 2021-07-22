import tensorflow as tf
import keras.backend as K

def mIoU(y_true, y_pred): # [bs,h,w,n_cl]
    """
    Mean Intersection over Union for categorical multi-class.
    """
    n_classes = K.int_shape(y_pred)[-1]
    true_pixels = K.argmax(y_true, axis=-1) # [bs,h,w]
    pred_pixels = K.argmax(y_pred, axis=-1)
    
    iou = []
    for i in range(n_classes): 
        true_labels = K.equal(true_pixels, i)
        pred_labels = K.equal(pred_pixels, i)
        inter = tf.cast(true_labels & pred_labels, tf.int32) # [bs,h,w]
        union = tf.cast(true_labels | pred_labels, tf.int32)
        iou_lab = K.sum(inter)/K.sum(union)
        iou.append(iou_lab)
    iou = tf.stack(iou)
    iou = tf.boolean_mask(iou, ~tf.math.is_nan(iou))
    
    return K.mean(iou)

def dice(y_true, y_pred, smooth=1):

    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)

    intersection = K.sum(y_true * y_pred, axis=[1,2])
    union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
    dice_coefx = 2.*K.mean((intersection + smooth)/(union + smooth), axis=0)
    
    return dice_coefx

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2])
    union =  K.sum(tf.cast(y_true, tf.int64), axis=[1,2]) + K.sum(y_pred, axis=[1,2])
    dice = 2.*K.mean((intersection + smooth)/(union + smooth), axis=0).numpy()
    return dice