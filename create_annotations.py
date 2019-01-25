import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
import os
import numpy as np
import cv2
import tensornets as nets
import time

def load(data_dir, min_shorter_side=None, max_longer_side=1000,
         batch_size=1, total_num=-1):
    assert cv2 is not None, '`load` requires `cv2`.'
    files = os.listdir(data_dir)
    if total_num !=-1:
      assert(total_num>0)
      files = files[:total_num]

    for batch_start in range(0, total_num, batch_size):
        x = cv2.imread("%s/JPEGImages/%s.ppm" % (data_dir, files[batch_start]))
        if min_shorter_side is not None:
            scale = float(min_shorter_side) / np.min(x.shape[:2])
        else:
            scale = 1.0
        if round(scale * np.max(x.shape[:2])) > max_longer_side:
            scale = float(max_longer_side) / np.max(x.shape[:2])
        x = cv2.resize(x, None, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_LINEAR)
        x = np.array([x], dtype=np.float32)
        scale = np.array([scale], dtype=np.float32)
        yield x, scale
        del x


def evaluate_class(ids, scores, boxes, annotations, files, ovthresh):
    if scores.shape[0] == 0:
        return 0.0, np.zeros(len(ids)), np.zeros(len(ids))

    # extract gt objects for this class
    diff = [np.array([obj['difficult'] for obj in annotations[filename]])
            for filename in files]
    total = sum([sum(x == 0) for x in diff])
    detected = dict(zip(files, [[False] * len(x) for x in diff]))

    # sort by confidence
    sorted_ind = np.argsort(-scores)
    ids = ids[sorted_ind]
    boxes = boxes[sorted_ind, :]

    # go down dets and mark TPs and FPs
    tp_list = []
    fp_list = []
    for d in range(len(ids)):
        actual = np.array([x['bbox'] for x in annotations[ids[d]]])
        difficult = np.array([x['difficult'] for x in annotations[ids[d]]])

        if actual.size > 0:
            iw = np.maximum(np.minimum(actual[:, 2], boxes[d, 2]) -
                            np.maximum(actual[:, 0], boxes[d, 0]) + 1, 0)
            ih = np.maximum(np.minimum(actual[:, 3], boxes[d, 3]) -
                            np.maximum(actual[:, 1], boxes[d, 1]) + 1, 0)
            inters = iw * ih
            overlaps = inters / (area(actual) + area(boxes[d, :]) - inters)
            jmax = np.argmax(overlaps)
            ovmax = overlaps[jmax]
        else:
            ovmax = -np.inf

        tp = 0.
        fp = 0.
        if ovmax > ovthresh:
            if difficult[jmax] == 0:
                if not detected[ids[d]][jmax]:
                    tp = 1.
                    detected[ids[d]][jmax] = True
                else:
                    fp = 1.
        else:
            fp = 1.
        tp_list.append(tp)
        fp_list.append(fp)

    tp = np.cumsum(tp_list)
    fp = np.cumsum(fp_list)
    recall = tp / float(total)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = np.mean([0 if np.sum(recall >= t) == 0
                  else np.max(precision[recall >= t])
                  for t in np.linspace(0, 1, 11)])

    return ap, precision, recall


def evaluate(results, data_dir, ovthresh=0.5, verbose=True):
    files = os.listdir(data_dir)
    files = files[:len(results)]
    annotations = load_annotations(data_dir, False)
    aps = []

    for c in range(20):
        ids = []
        scores = []
        boxes = []
        for (i, filename) in enumerate(files):
            pred = results[i][c]
            if pred.shape[0] > 0:
                for k in xrange(pred.shape[0]):
                    ids.append(filename)
                    scores.append(pred[k, -1])
                    boxes.append(pred[k, :4] + 1)
        ids = np.array(ids)
        scores = np.array(scores)
        boxes = np.array(boxes)
        _annotations = dict((k, v[c]) for (k, v) in annotations.iteritems())
        ap, _, _ = evaluate_class(ids, scores, boxes, _annotations,
                                  files, ovthresh)
        aps += [ap]

    strs = ''
    for c in range(20):
        strs += "| %6s " % classnames[c][:6]
    strs += '|\n'

    for ap in aps:
        strs += '|--------'
    strs += '|\n'

    for ap in aps:
        strs += "| %.4f " % ap
    strs += '|\n'

    strs += "Mean = %.4f" % np.mean(aps)
    return strs
    
def load_annotations(data_dir, train = True):

    columns = ['Filename', 'Roi.X1','Roi.Y1', 'Roi.X2', 'Roi.Y2','ClassId']
    annotations = pd.DataFrame(columns = columns)


    if train:
        annotations_path = os.path.join(data_dir, 'Train/Images')
        for dir_ in os.listdir(annotations_path):
            if dir_.startswith('.'):
               continue
            
            path = os.path.join(annotations_path, dir_, 'GT-' + dir_+ '.csv')
            annot_dir = pd.read_csv(path, sep=';')
            annot_dir['Filename'] = annot_dir['Filename'].apply(lambda path : annotations_path+'/'+dir_+'/'+path)
            annotations = pd.concat([annotations, annot_dir], axis = 0)
            
    else:
        path = os.path.join(data_dir, 'test', 'GT-final_test.test.csv')
        annotations = pd.read_csv(path, usecols = [ i for i in range(len(columns))], names = columns)


    
   # annotations = pd.read_csv(annotations_path, sep=';', 
    #                          usecols = [ i for i in range(len(columns))], names = columns)
    
    #annotations['Filename'] = annotations['Filename'].apply(lambda path : 'data/'+path)
    return shuffle(annotations)


def train_val_split(annotation , val_size = 0.1):

    n_total = annotation.shape[0]
    n_val = int(np.ceil(n_total * val_size))

    train_ann = annotation.copy()
    indexes = np.random.choice(train_ann.index, n_val)
    val_ann = train_ann.iloc[indexes]
    train_ann = train_ann.drop(indexes, axis = 0)
    val_ann = val_ann.reset_index()
    train_ann = train_ann.reset_index()
    return train_ann, val_ann

def load_data(annotations, batch_size = 128, target_size = (128,128), anchors = 5, classes = 43):
       
    feature_size = [x // 32 for x in target_size]
    cells = feature_size[0] * feature_size[1]
    nb_imgs, num_features = annotations.shape

    b = 0
    while True:
        if b == 0:
            idx = np.arange(nb_imgs)
        
        if b + batch_size > nb_imgs:
            b = 0
            yield None, None
        
        else:
            batch_num = batch_size


        imgs = np.zeros((batch_num,) + target_size + (3,), dtype=np.float32)
        probs = np.zeros((batch_num, cells, anchors, classes), dtype=np.float32)
        confs = np.zeros((batch_num, cells, anchors), dtype=np.float32)
        coord = np.zeros((batch_num, cells, anchors, 4), dtype=np.float32)
        proid = np.zeros((batch_num, cells, anchors, classes), dtype=np.float32)
        prear = np.zeros((batch_num, cells, 4), dtype=np.float32)
        areas = np.zeros((batch_num, cells, anchors), dtype=np.float32)
        upleft = np.zeros((batch_num, cells, anchors, 2), dtype=np.float32)
        botright = np.zeros((batch_num, cells, anchors, 2), dtype=np.float32)
        for i in range(batch_num):

            element = annotations.iloc[[i]]
            x = cv2.imread(element.iloc[0]['Filename'])
            h, w = x.shape[:2]
            cellx = 1. * w / feature_size[1]
            celly = 1. * h / feature_size[0]

            processed_objs = []
            
            centerx = .5 * (element.iloc[0]['Roi.X1'] + element.iloc[0]['Roi.X2'])
            centery = .5 * (element.iloc[0]['Roi.Y1'] + element.iloc[0]['Roi.Y2'])
            cx = centerx / cellx
            cy = centery / celly

            if (cx < feature_size[1] or cy < feature_size[0]):
                processed_object = [element.iloc[0]['ClassId'],
                                cx-np.floor(cx),
                                cy - np.floor(cy),
                                np.sqrt(float(element.iloc[0]['Roi.X2'] -element.iloc[0]['Roi.X1']) / w ),
                                np.sqrt(float(element.iloc[0]['Roi.Y2'] -element.iloc[0]['Roi.Y1']) / h ),
                                int(np.floor(cy)*feature_size[1]+np.floor(cx))]
            
            probs[i, processed_object[5], :, :] = [[0.] * classes] * anchors
            probs[i, processed_object[5], :, processed_object[0]] = 1.
            coord[i, processed_object[5], :, :] = [processed_object[1:5]] * anchors
            prear[i, processed_object[5], 0] = processed_object[1] - processed_object[3]**2 * .5 * feature_size[1]
            prear[i, processed_object[5], 1] = processed_object[2] - processed_object[4]**2 * .5 * feature_size[0]
            prear[i, processed_object[5], 2] = processed_object[1] + processed_object[3]**2 * .5 * feature_size[1]
            prear[i, processed_object[5], 3] = processed_object[2] + processed_object[4]**2 * .5 * feature_size[0]
            confs[i, processed_object[5], :] = [1.] * anchors            
            # Finalise the placeholders' values
            ul = np.expand_dims(prear[i, :, 0:2], 1)
            br = np.expand_dims(prear[i, :, 2:4], 1)
            wh = br - ul
            area = wh[:, :, 0] * wh[:, :, 1]
            upleft[i, :, :, :] = np.concatenate([ul] * anchors, 1)
            botright[i, :, :, :] = np.concatenate([br] * anchors, 1)
            areas[i, :, :] = np.concatenate([area] * anchors, 1)

            imgs[i] = cv2.resize(x, target_size,
                                interpolation=cv2.INTER_LINEAR)
        yield imgs, [probs, confs, coord, proid, areas, upleft, botright]
        b += batch_size



def main():
    anns = load_annotations('./DataCamp-project/Data/')
    anns1, anns2 = train_val_split(anns, 0.1)
    trains = load_data(anns)
    #print(anns.head())
    #cv2.imshow('aa',aa)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # Define a model
    inputs = tf.placeholder(tf.float32, [None, 128, 128, 3])
    is_training = tf.placeholder(tf.bool)
    model = nets.YOLOv2(inputs, nets.Darknet19, is_training=is_training, classes = 43)
    print("model loaded")
    # Define an optimizer
    step = tf.Variable(0, trainable=False)
    lr = tf.train.piecewise_constant(
        step, [100, 180, 320, 570, 1000, 40000, 60000],
        [1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-6, 1e-7])
    train = tf.train.MomentumOptimizer(lr, 0.9).minimize(model.loss,
                                                     global_step=step)
    print("Calculations initialized")
    with tf.Session() as sess:
    
    # Load Darknet19
        sess.run(tf.global_variables_initializer())
        sess.run(model.stem.pretrained())
        print("Darknet loaded")
    # Note that there are 16551 images (5011 in VOC07 + 11540 in VOC12).
    # When the mini-batch size is 48, 1 epoch consists of 344(=16551/48) steps.
    # Thus, 233 epochs will cover 80152 steps.
        losses = []
        print("Begin training")
        for i in range(10):

            # Iterate on VOC07+12 trainval once
            _t = time.time()
            for (imgs, metas) in trains:
                # `trains` returns None when it covers the full batch once
                
                if imgs is None:
                    break
                metas.insert(0, model.preprocess(imgs))  # for `inputs`
                metas.append(True)  # for `is_training`
                outs = sess.run([train, model.loss],
                                dict(zip(model.inputs, metas)))
                losses.append(outs[1])

        # Report step, learning rate, loss, weight decay, runtime
                print('***** %d %.5f %.5f %.5f %.5f *****' %
                    (sess.run(step), sess.run(lr),
                    losses[-1], sess.run(tf.losses.get_regularization_loss()),
                    time.time() - _t))
          
        #Test
        results = []
        tests = load('DataCamp-projet/Data/Test/Images', total_num=100)
        for (img, scale) in tests:
            outs = sess.run(model, {inputs: model.preprocess(img),
                                    is_training: False})
            results.append(model.get_boxes(outs, img.shape[1:3]))
        print(evaluate(results, 'DataCamp-projet/Data/Test/Images'))



if __name__ == '__main__':
    main()
