# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# For information on dataset see: https://github.com/mikacuy/pointnetvlad


import torchvision.transforms as transforms
import numpy as np
import random

from datasets.augmentation import *
from datasets.base_datasets import TrainingDataset
from datasets.freiburg.pnv_raw import PNVPointCloudLoader


class PNVTrainingDataset(TrainingDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_loader = PNVPointCloudLoader()


class TrainTransform:
    # Augmentations specific for PointNetVLAD datasets (RobotCar and Inhouse)
    def __init__(self, aug_mode):
        self.aug_mode = aug_mode
        if self.aug_mode == 1 or self.aug_mode == 2:
            # Augmentations without random rotation around z-axis
            
            t = [JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
                 RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4)]            
            """
          
            t = [JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
                 RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4), RandomRotation_mod(max_theta=20, p=0.5, axis=np.array([0, 0, 1]))]
            """
          
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 0: # No augmentation
            self.transform = None
        elif self.aug_mode == 3:
            t = [JitterPoints(sigma=0.001, clip=0.002)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 4:
            t = [RemoveRandomPoints(r=(0.0, 0.1))]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 5:
            t = [RandomTranslation(max_delta=0.01)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 6:
            # Augmentations with random rotation around z-axis
            t = [RemoveRandomBlock(p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 7:
            self.transform = None
        elif self.aug_mode == 8:
            self.transform = None
        elif self.aug_mode == 9:
            t = [RandomScale(min=0.9, max=1.1)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 10:
            t = [RandomShear()]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 11:
            t = [RandomFlip([0.25, 0.25, 0.])]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 12:
            t = [MoveRandomBlock(p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 13:
            t = [ElasticDistortion()]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 14:
            t = [RandomOcclusion(p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 15:
            t = [AddRandomNoise()]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 16:
            t = [RandomDropout()]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 17:
            t = [RandomTranslation(max_delta=0.1)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 18:
            t = [RandomTranslation(max_delta=0.5)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 19:
            # Augmentations with random rotation around z-axis
            t = [RemoveRandomBlock(p=1.1)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 20:
            t = [MoveRandomBlock(p=1.1)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 21:
            t = [RandomScale(min=0.5, max=1.5)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 22:
            t = [RemoveRandomBlock(p=1.1, scale=(0.02, 0.5), ratio=(0.3, 3.3))]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 23:
            t = [RandomTranslation(max_delta=0.25)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 24:
            t = [RemoveRandomPoints(r=(0.0, 0.5))]
            self.transform = transforms.Compose(t)

        elif self.aug_mode == 25:
            t = [RemoveRandomBlock(p=0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3))]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 26:
            t = [MoveRandomBlock(p=0.6)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 27:
            t = [MoveRandomBlock(p=0.6), RemoveRandomBlock(p=0.4)]
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 28:
            t = [MoveRandomBlock(p=0.6), RemoveRandomBlock(p=0.4), RemoveRandomPoints(r=(0.0, 0.5))]      
          
            self.transform = transforms.Compose(t)

        elif self.aug_mode == 29:
            t = [RemoveRandomBlock(p=0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock(p=0.8)]
            self.transform = transforms.Compose(t)  
        elif self.aug_mode == 30:
            t = [MoveRandomBlock(p=0.8)]          
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 31:
            t = [RemoveRandomBlock(p=0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3))]
            self.transform = transforms.Compose(t)


        elif self.aug_mode == 32:
            # Individually apply each augmentation with probability 0.5 to RemoveRandomBlock and the rest with probability 1.0
            t = [JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.5)),
                 RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4, scale=(0.02, 0.5))]       
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 33:
            t = [JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
                 RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4, scale=(0.02, 0.5))]    
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 34:
            # todos los efectos
            t = [JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.5)),
                 RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock(p=0.4), RandomScale(min=0.9, max=1.1)]    
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 35:
            # todos los efectos con p=0.4
            t = [JitterPoints(sigma=0.001, clip=0.002, probability=0.4), RemoveRandomPoints(r=(0.0, 0.5), p=0.4),
                 RandomTranslation(max_delta=0.01, p=0.4), RemoveRandomBlock(p=0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock(p=0.4), RandomScale(min=0.9, max=1.1, p=0.4)]    
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 36:
            t = [RemoveRandomBlock(p=0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock(p=1.1)]
            self.transform = transforms.Compose(t)  
        elif self.aug_mode == 37:
            # todos los efectos
            t = [RemoveRandomPoints(r=(0.0, 0.5)),
                 RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock(p=0.4), RandomScale(min=0.9, max=1.1)]    
            self.transform = transforms.Compose(t)
        elif self.aug_mode == 38:
            # todos los efectos con p=0.4
            t = [RemoveRandomPoints(r=(0.0, 0.5), p=0.4),
                 RandomTranslation(max_delta=0.01, p=0.4), RemoveRandomBlock(p=0.4, scale=(0.02, 0.5), ratio=(0.3, 3.3)), MoveRandomBlock(p=0.4), RandomScale(min=0.9, max=1.1, p=0.4)]    
            self.transform = transforms.Compose(t)
        
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e


