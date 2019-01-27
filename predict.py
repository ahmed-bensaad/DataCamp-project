import cv2
import tensorflow as tf
import tensornets as nets
from utils import darknet_preprocess

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2


img = nets.utils.load_img('./aaa.jpeg', target_size=128)
inputs = tf.placeholder(tf.float32, [None, 128, 128, 3])

with tf.variable_scope('yolo'):
    model1 = yolov2detector(inputs, nets.Darknet19, is_training=False, classes = 43)


with tf.Session() as sess:
    img1 = darknet_preprocess(img)  # equivalent to img = nets.preprocess(model, img)
    #_scope = tf.get_variable_scope().name
    #print(_scope)
    sess.run([w.assign(v) for (w, v) in zip(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='yolo'), np.load('weights_v2.npy'))])

    preds = sess.run(model1, {inputs: img1})

    boxes = model1.get_boxes(opts(),preds, img1.shape[1:3])
    for i,box in enumerate(boxes):

      if box.shape[0]>0:

        for boxx in list(box):

          bounds = boxx[:4]
          cv2.rectangle(img[0], (bounds[0],int(128 -bounds[1])),(bounds[2],int(128-bounds[3])),(0,255,0),1)
          cv2.putText(img[0],str(i), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)

cv2.imshow(img[0].astype(int))