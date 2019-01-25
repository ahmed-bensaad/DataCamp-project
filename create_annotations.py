import pandas as pd
from sklearn.utils import shuffle
import os
import numpy as np
import cv2



def load_annotations(data_dir, train = True):

    columns = ['Filename', 'Roi.X1','Roi.Y1', 'Roi.X2', 'Roi.Y2','ClassId']
    annotations = pd.DataFrame(columns = columns)


    if train:
        annotations_path = os.path.join(data_dir, 'train')
        for dir_ in os.listdir(annotations_path):
            path = os.path.join(annotations_path, dir_, 'GT-' + dir_+ '.csv')
            annot_dir = pd.read_csv(path, sep=';')
            annot_dir['Filename'] = annot_dir['Filename'].apply(lambda path : annotations_path+'/'+dir_+'/'+path)
            annotations = pd.concat([annotations, annot_dir], axis = 0, sort=True)
            
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

def load_data(annotations, batch_size = 64, target_size = (416,416), anchors = 5, classes = 62):
       
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
    anns = load_annotations('data/')
    anns1, anns2 = train_val_split(anns, 0.1)
    train = load_data(anns)
    print(anns.head())
    #cv2.imshow('aa',aa)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


if __name__=='__main__':
    main()
