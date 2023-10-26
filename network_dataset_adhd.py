#coding:utf8
import os
from torch.utils import data
import numpy as np
from sklearn.utils import shuffle
import nibabel as nib
import random
import pandas as pd
import glob
from sklearn.model_selection import StratifiedKFold
import math
import lmdb
import warnings
import h5py
from nilearn.connectome import ConnectivityMeasure
from sklearn.utils  import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
warnings.filterwarnings("ignore")
class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):

        self.input_size = input_size

        self.num_patches = self.input_size
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask # [196]

def make_train_test(length, fold_idx, seed = 0, ns_splits = 5):
    assert 0 <= fold_idx and fold_idx < 5, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=ns_splits, shuffle=True, random_state=seed)
    labels = np.zeros((length))

    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]
    return train_idx, test_idx

def mask_timeseries(timeser, mask = 30):
    rnd = np.random.random()
    if rnd < 0.2:
        return timeser
    
    time_len = timeser.shape[1]
    mask_index = np.array(random.sample(list(np.arange(0,time_len)),mask))
    bool_mask = np.zeros((time_len))
    bool_mask[mask_index]=1
    bool_mask = bool_mask.astype(bool)

    return timeser[:,~bool_mask]

def random_timeseries(timeser,sample_len):
    time_len = timeser.shape[1]
    st_thres = 1
    if time_len <= sample_len + st_thres:
        return timeser

    select_range = time_len - sample_len
    if select_range < 1:
        return timeser

    st = random.sample(list(np.arange(st_thres,select_range)),1)[0]
    return timeser[:,st:st+sample_len]

def slice_timeseries(timeser):
    time_len = timeser.shape[1]
    len_thres = int(time_len * 0.8)
    slices = []
    if time_len <= 30 + 2:
        return timeser
    for i in range(0,time_len - len_thres - 2):
        st = i + 2
        slices.append(timeser[:,st:st+len_thres])
    return np.stack(slices)

def mask_timeseries_per(timeser, mask = 30):
    rnd = np.random.random()
    if rnd < 0.2:
        return timeser

    time_len = timeser.shape[1]
    mask_len = int(mask * time_len /100)
    mask_index = np.array(random.sample(list(np.arange(0,time_len)),mask_len))
    bool_mask = np.zeros((time_len))
    bool_mask[mask_index]=1
    bool_mask = bool_mask.astype(bool)

    return timeser[:,~bool_mask]
class Task1DataRRP(data.Dataset):

    def __init__(self, root = None,mask_way='mask',mask_len=10, time_len=30):
        self.template = 'sch'
        self.root = root
        self.mask_way = mask_way
        self.mask_len = mask_len
        self.time_len = time_len

        self.names = [f for f in os.listdir(self.root) if self.template in f]

        print(f"Finding files: {len(self.names)}")
        self.correlation_measure = ConnectivityMeasure(kind='correlation')

    def __getitem__(self,index):
        name = self.names[index]
        img1 = np.load(os.path.join(self.root, name))
        if self.mask_way == 'mask':
            slices = [mask_timeseries(img1,mask=self.mask_len).T]
        else:
            slices = [img1.T]
        img1 = self.correlation_measure.fit_transform(slices)[0]
        img1[img1!=img1]=0

        target = 0

        rand = np.random.random()
        if rand <= 0.5:
            target = 0
        else:
            rand_name = np.random.choice(self.names)
            if name == rand_name:
                target = 0
            else:
                img2 = np.load(os.path.join(self.root, rand_name))
                if self.mask_way == 'mask':
                    img2 = self.correlation_measure.fit_transform([mask_timeseries(img2,mask=self.mask_len).T])[0]
                else:
                    img2 = self.correlation_measure.fit_transform([img2.T])[0]
                rois = np.random.choice(100, 50)
                
                target =1
                for roi in list(rois):
                    img1[roi] = img2[roi]

        onehot_lbl = np.zeros((2))
        onehot_lbl[target] = 1

        return img1,onehot_lbl

    def __len__(self):
        return len(self.names)

class Task1Data(data.Dataset):

    def __init__(self, root = None,mask_way='mask',mask_len=10, time_len=30):
        self.template = 'sch'
        self.root = root
        self.mask_way = mask_way
        self.mask_len = mask_len
        self.time_len = time_len

        self.names = [f for f in os.listdir(self.root) if self.template in f]

        print(f"Finding files: {len(self.names)}")
        self.correlation_measure = ConnectivityMeasure(kind='correlation')

    def __getitem__(self,index):
        name = self.names[index]
        img = np.load(os.path.join(self.root, name))
        if self.mask_way == 'mask':
            slices = [mask_timeseries(img,mask=self.mask_len).T, mask_timeseries(img,mask=self.mask_len).T]
            #slices = [img.T, mask_timeseries(img,mask=self.mask_len).T]
        elif self.mask_way == 'random':
            slices = [random_timeseries(img,sample_len=self.time_len).T, random_timeseries(img,sample_len=self.time_len).T]
        else:
            raise KeyError(f"mask way error, your input is {self.mask_way}")
        correlation_matrix = self.correlation_measure.fit_transform(slices)
        correlation_matrix[correlation_matrix!=correlation_matrix]=0
        return correlation_matrix[0], correlation_matrix[1]

    def __len__(self):
        return len(self.names)
        


class Task3Data(data.Dataset):

    def __init__(self, mask_way='mask',mask_len=10, time_len=30,shuffle_seed=42,is_train = True, is_test = False):
        self.template = 'sch'
        self.is_test = is_test
        self.is_train = is_train
        self.root = "/userhome/timeinvariant/run4_adhd/"

        self.mask_way = mask_way
        self.mask_len = mask_len
        self.time_len = time_len

        
        self.df = pd.read_csv("/userhome/timeinvariant/clean_adhd.csv")
        use_index = self.df[self.df['dx']!='pending'].index
        self.df = self.df.iloc[use_index].reset_index(drop=True)
        self.df['dx'] = self.df['dx'].astype(int)

        self.names = list(self.df['new_name'])
        self.df['dx'] = self.df['dx'].astype(int)

        all_data = np.array(self.names)
        lbls = np.array(list([0 if f == 0 else 1 for f in self.df['dx'] ]))
        sites = np.array(self.df['site'])

        train_length = int(len(self.df) * 0.7)
        val_length = int(len(self.df) * 0.15)
        test_length = int(len(self.df) * 0.15)

        split = StratifiedShuffleSplit(n_splits=1, test_size=val_length+test_length, train_size=train_length, random_state=42)
        for train_index, test_valid_index in split.split(all_data, sites):
            data_train, labels_train = all_data[train_index], lbls[train_index]
            data_rest, labels_rest = all_data[test_valid_index], lbls[test_valid_index]
            site_rest = sites[test_valid_index]

        split2 = StratifiedShuffleSplit(n_splits=1, test_size=test_length, random_state=shuffle_seed)
        for valid_index, test_index in split2.split(data_rest, site_rest):
            data_test, labels_test = data_rest[test_index], labels_rest[test_index]
            data_val, labels_val = data_rest[valid_index], labels_rest[valid_index]


        if is_test is True:
            print("Testing data:")
            self.imgs, self.lbls = data_test, labels_test
        elif is_train is True:
            print("Training data:")
            #self.imgs, self.lbls = data_train, labels_train
            self.imgs, self.lbls = np.concatenate([data_train, data_val],0), np.concatenate([labels_train, labels_val],0),
        else:
            print("Val data:")
            self.imgs, self.lbls = data_val, labels_val
        print(self.imgs.shape)
        self.correlation_measure = ConnectivityMeasure(kind='correlation')


    def __getitem__(self,index):
        name = self.imgs[index]
        lbl = self.lbls[index]
        img = np.load(os.path.join(self.root, f"{self.template}_{name}.npy"))
        if self.is_train is True:
            if self.mask_way == 'mask':
                slices = [mask_timeseries(img,self.mask_len).T,img.T]
            elif self.mask_way == 'mask_per':
                slices = [mask_timeseries_per(img,mask=self.mask_len).T, mask_timeseries_per(img,mask=self.mask_len).T]
            elif self.mask_way =='random':
                slices = [random_timeseries(img,self.time_len).T,img.T]
            else:
                slices = [img.T]
            correlation_matrix = self.correlation_measure.fit_transform(slices).mean(0)
        elif self.is_test is False:
            slices = [img.T]
            correlation_matrix = self.correlation_measure.fit_transform(slices)[0]
        else:
            slices = [img.T]
            correlation_matrix = self.correlation_measure.fit_transform(slices).mean(0)
        onehot_lbl = np.zeros((2))
        onehot_lbl[lbl] = 1
        correlation_matrix[correlation_matrix!=correlation_matrix]=0
        return correlation_matrix,onehot_lbl
    
    def __len__(self):
        return len(self.imgs)

class Task3Baseline(Task3Data):
    def __init__(self, shuffle_seed=42,is_train = True, is_test = False):
        self.template = 'sch'
        self.is_test = is_test
        self.is_train = is_train
        # self.root = "/data5/yang/large/abide1_timeseries"
        self.root = "/data5/yang/large/run4_adhd/"

        self.df = pd.read_csv("/data5/yang/large/clean_adhd.csv")
        use_index = self.df[self.df['dx']!='pending'].index
        self.df = self.df.iloc[use_index].reset_index(drop=True)
        self.df['dx'] = self.df['dx'].astype(int)

        self.names = list(self.df['new_name'])

        all_data = np.array(self.names)
        lbls = np.array(list([0 if f == 0 else 1 for f in self.df['dx'] ]))
        sites = np.array(self.df['site'])

        train_length = int(len(self.df) * 0.7)
        val_length = int(len(self.df) * 0.15)
        test_length = int(len(self.df) * 0.15)

        split = StratifiedShuffleSplit(n_splits=1, test_size=val_length+test_length, train_size=train_length, random_state=42)
        for train_index, test_valid_index in split.split(all_data, sites):
            data_train, labels_train = all_data[train_index], lbls[train_index]
            data_rest, labels_rest = all_data[test_valid_index], lbls[test_valid_index]
            site_rest = sites[test_valid_index]

        split2 = StratifiedShuffleSplit(n_splits=1, test_size=test_length, random_state=shuffle_seed)
        for valid_index, test_index in split2.split(data_rest, site_rest):
            data_test, labels_test = data_rest[test_index], labels_rest[test_index]
            data_val, labels_val = data_rest[valid_index], labels_rest[valid_index]


        if is_test is True:
            print("Testing data:")
            self.imgs, self.lbls = data_test, labels_test
        elif is_train is True:
            print("Training data:")
            self.imgs, self.lbls = data_train, labels_train
        else:
            print("Val data:")
            self.imgs, self.lbls = data_val, labels_val
        print(self.imgs.shape)
        self.correlation_measure = ConnectivityMeasure(kind='correlation')
    def __getitem__(self,index):
        name = self.imgs[index]
        lbl = self.lbls[index]
        img = np.load(os.path.join(self.root, f"{self.template}_{name}.npy"))
        slices = [img.T]
        correlation_matrix = self.correlation_measure.fit_transform(slices).mean(0)
        onehot_lbl = np.zeros((2))
        onehot_lbl[lbl] = 1
        correlation_matrix[correlation_matrix!=correlation_matrix]=0
        return correlation_matrix,onehot_lbl

class Task3DataFT(Task3Data):

    def __init__(self, shuffle_seed=42,is_train = True, is_test = False):
        self.template = 'sch'
        self.is_test = is_test
        self.is_train = is_train
        self.root = "/data5/yang/large/abide1_timeseries"
        
        self.df = pd.read_csv("/data5/yang/large/clean_abide1_train.csv")
        self.names = list(self.df['new_name'])
        all_data = np.array(self.names)
        lbls = np.array(list([1 if f == 1 else 0 for f in self.df['dx'] ]))
        sites = np.array(self.df['site'])

        train_length = int(len(self.df) * 0.7)
        val_length = int(len(self.df) * 0.15)
        test_length = int(len(self.df) * 0.15)

        train_index = self.df[self.df['is_train']==1].index
        rest_index = self.df[self.df['is_train']==0].index

        data_train = all_data[train_index]
        labels_train = lbls[train_index]

        data_rest = all_data[rest_index]
        site_rest = sites[rest_index]
        labels_rest = lbls[rest_index]

        print(data_rest.shape, site_rest.shape)

        split2 = StratifiedShuffleSplit(n_splits=1, test_size=test_length, random_state=shuffle_seed)
        for valid_index, test_index in split2.split(data_rest, site_rest):
            data_test, labels_test = data_rest[test_index], labels_rest[test_index]
            data_val, labels_val = data_rest[valid_index], labels_rest[valid_index]

        if is_test is True:
            print("Testing data:")
            #self.imgs, self.idx, self.lbls = self.make_subdataset(data_test, labels_test)
            self.imgs, self.lbls = data_test, labels_test
        elif is_train is True:
            print("Training data:")
            self.imgs, self.lbls = data_train, labels_train
            #
            #self.imgs,  self.lbls = self.make_subdataset(data_train, labels_train)
        else:
            print("Val data:")
            #self.imgs, self.idx, self.lbls = self.make_subdataset(data_val, labels_val)
            self.imgs, self.lbls = data_val, labels_val
        print(self.imgs.shape)
        self.correlation_measure = ConnectivityMeasure(kind='correlation')

class Task3DataSupCon(data.Dataset):

    def __init__(self, shuffle_seed=42,is_train = True, is_test = False):
        self.template = 'sch'
        self.is_test = is_test
        self.is_train = is_train
        self.root = "/data5/yang/large/abide1_timeseries"
        
        self.df = pd.read_csv("/data5/yang/large/clean_abide1.csv")
        self.names = list(self.df['new_name'])

        all_data = np.array(self.names)
        lbls = np.array(list([1 if f == 1 else 0 for f in self.df['dx'] ]))
        sites = np.array(self.df['site'])

        train_length = int(len(self.df) * 0.7)
        val_length = int(len(self.df) * 0.15)
        test_length = int(len(self.df) * 0.15)

        split = StratifiedShuffleSplit(n_splits=1, test_size=val_length+test_length, train_size=train_length, random_state=42)
        for train_index, test_valid_index in split.split(all_data, sites):
            data_train, labels_train = all_data[train_index], lbls[train_index]
            data_rest, labels_rest = all_data[test_valid_index], lbls[test_valid_index]
            site_rest = sites[test_valid_index]

        split2 = StratifiedShuffleSplit(n_splits=1, test_size=test_length, random_state=shuffle_seed)
        for valid_index, test_index in split2.split(data_rest, site_rest):
            data_test, labels_test = data_rest[test_index], labels_rest[test_index]
            data_val, labels_val = data_rest[valid_index], labels_rest[valid_index]

        if is_test is True:
            print("Testing data:")
            #self.imgs, self.idx, self.lbls = self.make_subdataset(data_test, labels_test)
            self.imgs, self.lbls = data_test, labels_test
        elif is_train is True:
            print("Training data:")
            self.imgs, self.lbls = data_train, labels_train
            #
            #self.imgs,  self.lbls = self.make_subdataset(data_train, labels_train)
        else:
            print("Val data:")
            #self.imgs, self.idx, self.lbls = self.make_subdataset(data_val, labels_val)
            self.imgs, self.lbls = data_val, labels_val
        print(self.imgs.shape)
        self.correlation_measure = ConnectivityMeasure(kind='correlation')


    def __getitem__(self,index):
        name = self.imgs[index]
        lbl = self.lbls[index]
        img = np.load(os.path.join(self.root, f"{self.template}_{name}.npy"))
        time_len = 15
        if self.is_train is True:
            slices = [mask_timeseries(img,10).T,mask_timeseries(img,10).T]
            correlation_matrix = self.correlation_measure.fit_transform(slices)
        elif self.is_test is False:
            slices = [img.T]
            correlation_matrix = self.correlation_measure.fit_transform(slices)[0]
        else:
            slices = [img.T]
            correlation_matrix = self.correlation_measure.fit_transform(slices)[0]
        onehot_lbl = np.zeros((2))
        onehot_lbl[lbl] = 1
        correlation_matrix[correlation_matrix!=correlation_matrix]=0
        return correlation_matrix[0], correlation_matrix[1],onehot_lbl

    
    def __len__(self):
        return len(self.imgs)

class Task3DataNetwork(data.Dataset):

    def __init__(self, shuffle_seed,is_train = True, is_test = False):
        self.template = 'ho'
        self.root = "//yang/large/abide1_timeseries"
        df = pd.read_csv("/data5/yang/large/clean_abide1.csv")
        all_data = np.array(list(df['new_name']))
        lbls = np.array(list([1 if f == 1 else 0 for f in df['dx'] ]))
        sites = np.array(df['site'])

        train_length = int(len(df) * 0.65)
        val_length = int(len(df) * 0.15)
        test_length = int(len(df) * 0.2)

        split = StratifiedShuffleSplit(n_splits=1, test_size=val_length+test_length, train_size=train_length, random_state=42)
        for train_index, test_valid_index in split.split(all_data, sites):
            data_train, labels_train = all_data[train_index], lbls[train_index]
            data_rest, labels_rest = all_data[test_valid_index], lbls[test_valid_index]
            site_rest = sites[test_valid_index]

        split2 = StratifiedShuffleSplit(n_splits=1, test_size=test_length, random_state=shuffle_seed)
        for valid_index, test_index in split2.split(data_rest, site_rest):
            data_test, labels_test = data_rest[test_index], labels_rest[test_index]
            data_val, labels_val = data_rest[valid_index], labels_rest[valid_index]

        if is_test is True:
            print("Testing data:")
            self.imgs = data_test
            self.lbls = labels_test
        elif is_train is True:
            print("Training data:")
            self.imgs = data_train
            self.lbls = labels_train
        else:
            print("Val data:")
            self.imgs = data_val
            self.lbls = labels_val

        self.cm = ConnectivityMeasure(kind='correlation')
        print(f"Finding files: {len(self.imgs)}")
        print(self.imgs.shape)
        print(self.lbls.shape)

    def __getitem__(self,index):
        img = np.array(np.load(os.path.join(self.root, f"{self.template}_{self.imgs[index]}.npy")))
        fc = self.cm.fit_transform(img.T[None])[0]
        fc[fc!=fc]=0
        lbl = self.lbls[index]
        onehot_lbl = np.zeros((2))
        onehot_lbl[lbl] = 1
        return fc,onehot_lbl
    
    def __len__(self):
        return len(self.imgs)