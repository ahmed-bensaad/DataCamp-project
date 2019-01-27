import numpy as np 
from tensornets.references.darkflow_utils.get_boxes import yolov2_box
import tensorflow as tf
import cv2
from functools import reduce

def get_v2_boxes(opts, outs, source_size, threshold=0.1):
    h, w = source_size
    boxes = [[] for _ in range(opts['classes'])]
    opts['thresh'] = threshold
    results = yolov2_box(opts, np.array(outs[0], dtype=np.float32))
    for b in results:
        idx, box = parse_box(b, threshold, w, h)
        if idx is not None:
            boxes[idx].append(box)
    for i in range(opts['classes']):
        boxes[i] = np.asarray(boxes[i], dtype=np.float32)
    return boxes


def local_flatten(x, kernel_size, name=None):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    assert isinstance(kernel_size, tuple)
    x = [[tf.strided_slice(x, (0, i, j), tf.shape(x)[:-1], (1,) + kernel_size)
          for j in range(kernel_size[1])] for i in range(kernel_size[0])]
    return tf.concat(reduce(lambda x, y: x + y, x), axis=-1, name=name)

def parse_scopes(inputs):
    if not isinstance(inputs, list):
        inputs = [inputs]
    outputs = []
    for scope_or_tensor in inputs:
        if isinstance(scope_or_tensor, tf.Tensor):
            outputs.append(scope_or_tensor.aliases[0])
        elif isinstance(scope_or_tensor, str):
            outputs.append(scope_or_tensor)
        else:
            outputs.append(None)
    return outputs


def darknet_preprocess(x, target_size=(128,128)):
    # Refer to the following darkflow
    # https://github.com/thtrieu/darkflow/blob/master/darkflow/net/yolo/predict.py
    if target_size is None or target_size[0] is None or target_size[1] is None:
        y = x.copy()
    else:
        h, w = target_size
        assert cv2 is not None, 'resizing requires `cv2`.'
        y = np.zeros((len(x), h, w, x.shape[3]))
        for i in range(len(x)):
            y[i] = cv2.resize(x[i], (w, h), interpolation=cv2.INTER_CUBIC)
    y = y[:, :, :, ::-1]
    y /= 255.
    return y
