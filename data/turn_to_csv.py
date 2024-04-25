import os
import pandas as pd

path = '/data1/lht/medical_image_seg/MedSegDiff-master/data/ISIC'
train_data_dir = 'ISBI2016_ISIC_Part3B_Test_Data'
train_gt_dir = 'ISBI2016_ISIC_Part3B_Test_Data'
csv_path_train = '/data1/lht/medical_image_seg/MedSegDiff-master/data/ISIC/ISBI2016_ISIC_Part3B_Test_GroundTruth.csv'

col_names = ['img','seg']
df = pd.DataFrame()

for (root,dirs,files) in os.walk(os.path.join(path,train_data_dir), topdown=True):
    for name in files:
        preind = name.split('.')[0]
        if not preind.endswith('n'):

            segname = preind + '_Segmentation.png'
            segpath = os.path.join( train_gt_dir, segname)
            imgpath = os.path.join( train_data_dir, name)
            data = pd.DataFrame([[imgpath,segpath]], columns = col_names)
            df = df._append(data, ignore_index=True)
            df.to_csv(csv_path_train)