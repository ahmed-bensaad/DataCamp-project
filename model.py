import tensorflow as tf 

from functools import reduce
from tensornets.layers import batch_norm ,bias_add, conv2d,darkconv, dropout,flatten,fc
from utils import local_flatten

def get_weights(scope=None):
    scope = parse_scopes(scope)[0]
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

def opts():

    opt = {'anchors': [0.57273, 0.677385, 1.87446, 2.06253, 3.33843,

                               5.47434, 7.88282, 3.52778, 9.77052, 9.16828]}

    opt.update({'num': len(opt['anchors']) // 2})

    opt.update({'classes': 43, 'labels': list(np.arange(43))})

    return opt

def yolov2detector(x, stem_fn, stem_out=None, is_training=False, classes=43,
           scope='YOLOV2', reuse=False):

    inputs = x
    opt = opts()
    stem = x = stem_fn(x, is_training, stem=True, scope='stem')
    p = x.p
    if stem_out is not None:
        stem = x = remove_head(x, stem_out)
    x = darkconv(x, 1024, 3, scope='conv7')
    x = darkconv(x, 1024, 3, scope='conv8')
    p = darkconv(p, 64, 1, scope='conv5a')
    p = local_flatten(p, 2, name='flat5a')
    x = tf.concat([p, x], axis=3, name='concat')
    x = darkconv(x, 1024, 3, scope='conv9')
    x = darkconv(x, (classes + 5) * 5, 1, onlyconv=True, scope='linear')
    x.aliases = []
    def get_boxes(opts, outs, source_size, threshold=0.1):
        return get_v2_boxes(opts, outs, source_size, threshold=0.1)
    x.get_boxes = get_boxes
    x.stem = stem
    x.inputs = [inputs]
    x.inputs += v2_inputs(x.shape[1:3], opt['num'], classes, x.dtype)
    if isinstance(is_training, tf.Tensor):
        x.inputs.append(is_training)
    x.loss = v2_loss(x, opt['anchors'], classes)

    return x

