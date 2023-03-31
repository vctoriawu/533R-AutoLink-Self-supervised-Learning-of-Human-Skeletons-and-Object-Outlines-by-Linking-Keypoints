import torch
import torch.utils.data
import torchvision
from torchvision import transforms
import os
import random
import ast
import cv2
import pickle
import bz2
import scipy.io as sio
import pandas as pd
import numpy as np
from abc import ABC
from torch.utils.data import Dataset

class TrainSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        data_dir = os.path.join(data_root, "LV_cleaned")

        mode = "train"
        # Read the data CSV file
        self.data_info = pd.read_csv(os.path.join(data_root, "files/lv_plax2_cleaned_info_landmark_gt_filtered.csv"))

        # Rename the index column so the processed data can be tracked down later
        self.data_info = self.data_info.rename(columns={'Unnamed: 0': 'db_indx'})

        # Get the correct data split. Data split was applied during the preprocessing
        self.data_info = self.data_info[self.data_info.split == mode]

        #if logger is not None:
        #    logger.info(f'#{mode}: {len(self.data_info)}')

        # Add root directory to file names to create full path to data
        self.data_info['cleaned_path'] = self.data_info.apply(lambda row: os.path.join(data_dir, row['file_name']),
                                                              axis=1)

        #self.imgs = torchvision.datasets.ImageFolder(root=data_root, transform=self.transform)

    def __getitem__(self, idx):
        
        data_item = {}
        # Get the data at index
        data = self.data_info.iloc[idx]

        # Unpickle the data
        pickle_file = bz2.BZ2File(data['cleaned_path'], 'rb')
        mat_contents = pickle.load(pickle_file)
        cine = mat_contents['resized'] # 224x224xN

        # Extracting the ED frame
        if data['d_frame_number'] > cine.shape[-1]:
            ed_frame = cine[:, :, -1]
        else:
            ed_frame = cine[:, :, data['d_frame_number']-1]

        # Create a new 3-channel image with the same size
        new_image = np.zeros((3, 224, 224), dtype=np.uint8)

        # Duplicate the grayscale channel into the other two channels
        new_image[0, :, :] = ed_frame
        new_image[1, :, :] = ed_frame
        new_image[2, :, :] = ed_frame

        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert NumPy array to PIL Image object
            #transforms.Resize((128, 128)),  # Resize the PIL Image object
            transforms.CenterCrop(128),
            transforms.ToTensor()  # Convert PIL Image object to tensor
        ])
        
        new_image = transform(np.transpose(new_image, (1, 2, 0)))
        sample = {'img': np.transpose(new_image, (0, 1, 2))}
        return sample

    def __len__(self):
        return len(self.data_info)


class TrainRegSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        data_dir = os.path.join(data_root, "LV_cleaned")
        mode = "val"

        # Read the data CSV file
        self.data_info = pd.read_csv(os.path.join(data_root, "files/lv_plax2_cleaned_info_landmark_gt_filtered.csv"))

        # Rename the index column so the processed data can be tracked down later
        self.data_info = self.data_info.rename(columns={'Unnamed: 0': 'db_indx'})

        # Get the correct data split. Data split was applied during the preprocessing
        self.data_info = self.data_info[self.data_info.split == mode]

        #if logger is not None:
        #    logger.info(f'#{mode}: {len(self.data_info)}')

        # Add root directory to file names to create full path to data
        self.data_info['cleaned_path'] = self.data_info.apply(lambda row: os.path.join(data_dir, row['file_name']),
                                                              axis=1)

        #self.imgs = torchvision.datasets.ImageFolder(root=data_root, transform=self.transform)

    def __getitem__(self, idx):
                
        data_item = {}
        # Get the data at index
        data = self.data_info.iloc[idx]

        # Unpickle the data
        pickle_file = bz2.BZ2File(data['cleaned_path'], 'rb')
        mat_contents = pickle.load(pickle_file)
        cine = mat_contents['resized'] # 224x224xN

        # Extracting the ED frame
        if data['d_frame_number'] > cine.shape[-1]:
            ed_frame = cine[:, :, -1]
        else:
            ed_frame = cine[:, :, data['d_frame_number']-1]

        # Create a new 3-channel image with the same size
        new_image = np.zeros((3, 224, 224), dtype=np.uint8)

        # Duplicate the grayscale channel into the other two channels
        new_image[0, :, :] = ed_frame
        new_image[1, :, :] = ed_frame
        new_image[2, :, :] = ed_frame

        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert NumPy array to PIL Image object
            #transforms.Resize((128, 128)),  # Resize the PIL Image object
            transforms.CenterCrop(128),
            transforms.ToTensor()  # Convert PIL Image object to tensor
        ])
        
        new_image = transform(np.transpose(new_image, (1, 2, 0)))        
        sample = {'img': np.transpose(new_image, (0, 1, 2)), 'keypoints': torch.tensor(0)}
        return sample

    def __len__(self):
        return len(self.data_info)


class TestSet(torch.utils.data.Dataset):
    def __init__(self, data_root, image_size):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        data_dir = os.path.join(data_root, "LV_cleaned")

        mode = "test"
        # Read the data CSV file
        self.data_info = pd.read_csv(os.path.join(data_root, "files/lv_plax2_cleaned_info_landmark_gt_filtered.csv"))

        # Rename the index column so the processed data can be tracked down later
        self.data_info = self.data_info.rename(columns={'Unnamed: 0': 'db_indx'})

        # Get the correct data split. Data split was applied during the preprocessing
        self.data_info = self.data_info[self.data_info.split == mode]

        #if logger is not None:
        #    logger.info(f'#{mode}: {len(self.data_info)}')

        # Add root directory to file names to create full path to data
        self.data_info['cleaned_path'] = self.data_info.apply(lambda row: os.path.join(data_dir, row['file_name']),
                                                              axis=1)

        #self.imgs = torchvision.datasets.ImageFolder(root=data_root, transform=self.transform)

    def __getitem__(self, idx):
                
        data_item = {}
        # Get the data at index
        data = self.data_info.iloc[idx]

        # Unpickle the data
        pickle_file = bz2.BZ2File(data['cleaned_path'], 'rb')
        mat_contents = pickle.load(pickle_file)
        cine = mat_contents['resized'] # 224x224xN

        # Extracting the ED frame
        if data['d_frame_number'] > cine.shape[-1]:
            ed_frame = cine[:, :, -1]
        else:
            ed_frame = cine[:, :, data['d_frame_number']-1]
        
        # Create a new 3-channel image with the same size
        new_image = np.zeros((3, 224, 224), dtype=np.uint8)

        # Duplicate the grayscale channel into the other two channels
        new_image[0, :, :] = ed_frame
        new_image[1, :, :] = ed_frame
        new_image[2, :, :] = ed_frame

        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert NumPy array to PIL Image object
            #transforms.Resize((128, 128)),  # Resize the PIL Image object
            transforms.CenterCrop(128),
            transforms.ToTensor()  # Convert PIL Image object to tensor
        ])
        
        new_image = transform(np.transpose(new_image, (1, 2, 0)))

        sample = {'img': np.transpose(new_image, (0, 1, 2)), 'keypoints': torch.tensor(0)}
        return sample

    def __len__(self):
        return len(self.data_info)


def test_epoch_end(batch_list_list):
    raise NotImplementedError()
