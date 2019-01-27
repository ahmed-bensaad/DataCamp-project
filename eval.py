import tensorflow as tf
import numpy as np 
from model import yolov2detector
from data_gens import load
import tensornets as nets 
from utils import darknet_preprocess


def evaluate_class(ids, scores, boxes, annotations, files, ovthresh):
    if scores.shape[0] == 0:
        return 0.0, np.zeros(len(ids)), np.zeros(len(ids))

    # extract gt objects for this class
    #diff = [np.array([obj['difficult'] for obj in annotations[filename]])
    #        for filename in files]
    #total = sum([sum(x == 0) for x in diff])
    detected = dict(zip(files, [[False] * len(x) for x in annotations['Filename']]))
    # sort by confidence
    sorted_ind = np.argsort(-scores)
    ids = ids[sorted_ind]
    boxes = boxes[sorted_ind, :]

    # go down dets and mark TPs and FPs
    tp_list = []
    fp_list = []
    for d in range(len(ids)):
        
      #  difficult = np.array([x['difficult'] for x in annotations[ids[d]]])
        filename = ids[d]
        rows = annotations.loc[annotations['Filename']==filename]
        actual = np.zeros((rows.shape[1],4))
        for idxx, row in rows.iterrows():
            bbox = np.array([row['Roi.X1'],row['Roi.Y1'],row['Roi.X2'],row['Roi.Y2']])
            actual[idxx, : ] = bbox
        if np.sum(actual) != 0:
            iw = np.maximum(np.minimum(actual[:, 2], boxes[d, 2]) -
                            np.maximum(actual[:, 0], boxes[d, 0]) + 1, 0)
            ih = np.maximum(np.minimum(actual[:, 3], boxes[d, 3]) -
                            np.maximum(actual[:, 1], boxes[d, 1]) + 1, 0)
            inters = iw * ih
            overlaps = inters / (area(actual) + area(boxes[d, :]) - inters) #overlaps = IoU
            jmax = np.argmax(overlaps)
            ovmax = overlaps[jmax]
        else:
            ovmax = -np.inf

        tp = 0.
        fp = 0.
        if ovmax > ovthresh:
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
    recall = tp / float(len(ids))
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = np.mean([0 if np.sum(recall >= t) == 0
                  else np.max(precision[recall >= t])
                  for t in np.linspace(0, 1, 11)])

    return ap, precision, recall


def evaluate(results, data_dir, ovthresh=0.5, verbose=True):
    files = os.listdir(os.path.join(data_dir,'Train','Images'))
    print(files)
    files = files[:len(results)]
    annotations = load_annotations(data_dir, True)
    aps = []

    for c in range(43):
        ids = []
        scores = []
        boxes = []
        for (i, filename) in enumerate(files):
            pred = results[i][c]
            if pred.shape[0] > 0:
                for k in range(pred.shape[0]):
                    ids.append(filename)
                    scores.append(pred[k, -1])
                    boxes.append(pred[k, :4] + 1)
                print(scores)
        
        ids = np.array(ids)
        scores = np.array(scores)
        boxes = np.array(boxes)
        #_annotations = dict((k, v[c]) for (k, v) in annotations.iteritems())
        c_annot = annotations.loc[annotations['ClassId']==c]
        ap, precision, recall = evaluate_class(ids, scores, boxes, c_annot,
                                  files, ovthresh)
        aps += [ap]
    classnames = np.arange(43)
    strs = ''
    for c in range(43):
        strs += "|"+str( classnames[c])#[:6]
    strs += '|\n'

    for ap in aps:
        strs += '|--------'
    strs += '|\n'

    for ap in aps:
        strs += "| %.4f " % ap
    strs += '|\n'

    strs += "Mean = %.4f" % np.mean(aps)
    print('final precision : {}'.format(precision))
    print('final recall : {}'.format(recall))
    return strs



def eval():
    inputs = tf.placeholder(tf.float32, [None, 128, 128, 3])

    with tf.variable_scope('yolo'):
        model1 = yolov2detector(inputs, nets.Darknet19, is_training=False, classes = 43)

    with tf.Session() as sess:

        sess.run([w.assign(v) for (w, v) in zip(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='yolo'), np.load('weights_v2.npy'))])
        results = []
        tests = load('./DataCamp-project/Data/', total_num=-1)
        #tests = load_data(anns)
        for (img, scale) in tests:
            outs = sess.run(model1, {inputs: darknet_preprocess(img),
                                    is_training: False})

            result = model1.get_boxes(opts(),outs, img.shape[1:3])
            results.append(result)
        print(evaluate(results, './DataCamp-project/Data/'))


if __name__== '__main__':
    eval()