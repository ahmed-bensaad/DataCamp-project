import os
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



columns = ['Filename', 'ClassId']
#Combine all train annotations into one table 
def load_annotations(data_dir):

    
    annotations = pd.DataFrame(columns = columns)


    annotations_path = os.path.join(data_dir, 'data/Train')

    #iterate over all classes folders to get the class annotations
    for dir_ in os.listdir(annotations_path):
        if dir_.startswith('.'):
           continue

        path = os.path.join(annotations_path, dir_, 'GT-' + dir_+ '.csv')
        annot_dir = pd.read_csv(path, sep=';', usecols=columns)
        annot_dir['Filename'] = annot_dir['Filename'].apply(lambda path : annotations_path+'/'+dir_+'/'+path)
        annotations = pd.concat([annotations, annot_dir], axis = 0)

    return shuffle(annotations)


#Select Filename (entire paths) and ClassId
def test_ann(path_csv):
    filename = 'GT-final_test.csv'
    ann = pd.read_csv(os.path.join(path_csv,filename), sep =';', usecols=columns)
    ann['Filename'] = ann['Filename'].apply(lambda img : path_csv+'/'+img)
    return ann
    
def main():
    #create the whole data annotations file
    # We will save the test csv in the same 
    train_annotations = load_annotations('.')
    test_annotations = test_ann('./data/Test')
    print(test_annotations.head())
    print(train_annotations.head())
    # write csv
    train_annotations.to_csv('data/Train.csv')
    test_annotations.to_csv('data/Test.csv')
    print('Annotations generated ! ')

if __name__=='__main__':
    main()