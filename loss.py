from keras import backend as K

def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)
    weights /= K.sum(weights, keepdims=True)
    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        loss = weights * cross_entropy
        loss = K.mean(loss, -1)
        return loss
    return loss


def categorical_focal_loss(gamma=2., alpha=1):
    def categorical_focal_loss_fixed(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(loss, -1)
    return categorical_focal_loss_fixed


def weighted_categorical_focal_loss(weights, gamma=2., alpha=1):
    weights = K.variable(weights)
    weights /= K.sum(weights, keepdims=True)
    def categorical_focal_loss_fixed(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(loss, -1)
    return categorical_focal_loss_fixed


def focal_n_dice_loss(gamma=2., alpha=1):

    def categorical_focal_loss_fixed(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        def dice_loss(y_true, y_pred):
            numerator = 2 * K.sum(y_true * y_pred, axis=(1,2,3))
            denominator = K.sum(y_true + y_pred, axis=(1,2,3))

            return K.reshape(1 - numerator / denominator, (-1, 1, 1))

        return K.mean(loss, -1) + dice_loss(y_true, y_pred)

    return categorical_focal_loss_fixed


def bce_n_dice_loss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        numerator = 2 * K.sum(y_true * y_pred, axis=(1,2,3))
        denominator = K.sum(y_true + y_pred, axis=(1,2,3))
        return K.reshape(1 - numerator / denominator, (-1, 1, 1))
    return K.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
