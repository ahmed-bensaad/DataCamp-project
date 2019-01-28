import os
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def load_annotations(data_dir):

    columns = ['Filename', 'ClassId']
    annotations = pd.DataFrame(columns = columns)


    annotations_path = os.path.join(data_dir, 'data')

    #iterate over all classes folders to get the class annotations
    for dir_ in os.listdir(annotations_path):
        if dir_.startswith('.'):
           continue

        path = os.path.join(annotations_path, dir_, 'GT-' + dir_+ '.csv')
        annot_dir = pd.read_csv(path, sep=';', usecols=['Filename','ClassId'])
        annot_dir['Filename'] = annot_dir['Filename'].apply(lambda path : annotations_path+'/'+dir_+'/'+path)
        annotations = pd.concat([annotations, annot_dir], axis = 0)
    
    return shuffle(annotations)


    
def main():
    #create the whole data annotations file
    annotations = load_annotations('.')

    #train/test split

    data_train, data_test = train_test_split(annotations, test_size=0.3)
    
    # write csv
    data_train.to_csv('data/Train.csv')
    data_test.to_csv('data/Test.csv')
    print('Annotations generated ! ')

if __name__=='__main__':
    main()