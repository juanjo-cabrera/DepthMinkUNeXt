import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch.nn.functional as F
import torch
from operator import itemgetter
import time
from config import PARAMS
from PIL import Image
import open3d as o3d
import os
import MinkowskiEngine as ME
from datasets.freiburg.pnv_raw import PNVPointCloudLoader


class PCDDataset(Dataset):
    def __init__(self, FolderDataset, transform=None, should_invert=True):
        self.FolderDataset = FolderDataset
        self.transform = transform
        self.should_invert = should_invert
        self.pc_loader = PNVPointCloudLoader()

    def __getitem__(self, index):
        def coordenadas(ruta):
           x_index = ruta.index('_x')
           y_index = ruta.index('_y')
           a_index = ruta.index('_a')
           x = ruta[x_index+2:y_index]
           y = ruta[y_index+2:a_index]
           coor_list = [x,y]
           coor = torch.from_numpy(np.array(coor_list, dtype=np.float32))
           return coor
        
        samples_root = self.FolderDataset.samples[index]
        file_pathname = samples_root[0]
        query = self.pc_loader.read_pc(file_pathname)
        
        query_points = torch.tensor(query['points'], dtype=torch.float)      
        query_color = torch.tensor(query['colors'], dtype=torch.float)
        # normalize the color values /255.0
        query_color = query_color / 255.0

        if self.transform is not None:
            query_points = self.transform(query_points)

        query_pc = {}
        query_pc['points'] = query_points
        query_pc['colors'] = query_color
    
        coor = coordenadas(samples_root[0])
        return query_pc, coor

    def __len__(self):
        return len(self.FolderDataset.samples)
    

class FreiburgPCDMap():
    def __init__(self, map_dset, model):
        self.map_dset = map_dset
        self.pc_loader = PNVPointCloudLoader()
        self.get_whole_map()
        self.compute_whole_vectors(model)
        

    def get_coordinates(self, samples_tuple):
        map_coordinates = []
        for sample_tuple in samples_tuple:
            ruta = sample_tuple[0]
            x_index = ruta.index('_x')
            y_index = ruta.index('_y')
            a_index = ruta.index('_a')
            x=ruta[x_index+2:y_index]
            y=ruta[y_index+2:a_index]
            coor_list= [x,y]
            coor = torch.from_numpy(np.array(coor_list,dtype=np.float32))
            map_coordinates.append(coor)
        return map_coordinates
    
    
    def get_building_map(self):
        samples_root = self.map_dset.samples
        room_coordinates = self.get_coordinates(samples_root)
        pcds = []

        for sample_tuple in samples_root:
            file_pathname = sample_tuple[0]
    
            query = self.pc_loader.read_pc(file_pathname)
        
            query_points = torch.tensor(query['points'], dtype=torch.float)      
            query_color = torch.tensor(query['colors'], dtype=torch.float)
            # normalize the color values /255.0
            query_color = query_color / 255.0

            #if self.transform is not None:
            #    query_points = self.transform(query_points)

            query_pc = {}
            query_pc['points'] = query_points
            query_pc['colors'] = query_color
            
            pcds.append(query_pc)

        return pcds, room_coordinates
    
    def get_whole_map(self):
        self.pcds, self.pcds_coordinates = self.get_building_map()
   


    def get_latent_vector(self, pcd, model):
        device = torch.device(PARAMS.cuda_device if torch.cuda.is_available() else 'cpu')
        # set cuda device
        torch.cuda.set_device(device)
        points = pcd['points']
        color = pcd['colors']

        points = points.detach().cpu().numpy()
        color = color.detach().cpu().numpy()
        coords, feats = ME.utils.sparse_quantize(coordinates=points, features=color, quantization_size=PARAMS.quantization_size)
        #coords = ME.utils.sparse_quantize(coordinates=test_points,
        #                                  quantization_size=0.01).cuda()

        bcoords = ME.utils.batched_coordinates([coords])
        if PARAMS.use_rgb:
            feats = torch.cat(feats, dim=0)
        else:
            feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)


        #coords = ME.utils.sparse_quantize(coordinates=pcd,
        #                                      quantization_size=PARAMS.quantization_size).cuda()

        #bcoords = ME.utils.batched_coordinates([coords])
        #feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32).cuda()
        batch = {'coords': bcoords.to(device), 'features': feats.to(device)}

        latent_vector = model(batch)['global'].detach().cpu().numpy()
        return latent_vector

    def compute_whole_vectors(self, model):
        map_vectors = []
        for pcd in self.pcds:         
            latent_vector = self.get_latent_vector(pcd, model)
            map_vectors.append(latent_vector)
        self.map_vectors = map_vectors


    def evaluate_error_position(self, test_vector, coor_test):
        start_time = time.time()

        distances = []
        test_vector = torch.from_numpy(test_vector)
        for vector in self.map_vectors:
            # print('Vector size: ', vector.shape[0])
            # print(f'Memory size of a vector: {vector.element_size() * vector.nelement()} Bytes')
            # convert to tensor to calculate the distance
            #vector = torch.tensor(vector) # Could not infer dtype of builtin_function_or_method
            vector = torch.from_numpy(vector)            
            euclidean_distance = F.pairwise_distance(test_vector, vector, keepdim=True)
            distances.append(euclidean_distance)
        ind_min = distances.index(min(distances))

        coor_map = self.pcds_coordinates[ind_min]
        end_time = time.time()
        processing_time = end_time - start_time
        # print(f'Processing time: {processing_time}')
        error_localizacion = F.pairwise_distance(coor_test, coor_map.cuda())
        return error_localizacion.detach().cpu().numpy(), processing_time



def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()


map_data = dset.DatasetFolder(root=PARAMS.map_dir, loader=PNVPointCloudLoader, extensions='.ply')
map_dataloader = DataLoader(map_data,
                        shuffle=False,
                        num_workers=0,
                        batch_size=1)


test_data_cloudy = dset.DatasetFolder(root=PARAMS.testing_cloudy_dir, loader=PNVPointCloudLoader, extensions='.ply')
test_data_night = dset.DatasetFolder(root=PARAMS.testing_night_dir, loader=PNVPointCloudLoader, extensions='.ply')
test_data_sunny = dset.DatasetFolder(root=PARAMS.testing_sunny_dir, loader=PNVPointCloudLoader, extensions='.ply')

test_pcd_dataset_cloudy = PCDDataset(test_data_cloudy)
test_pcd_dataloader_cloudy = DataLoader(test_pcd_dataset_cloudy,
                        shuffle=False,
                        num_workers=0,
                        batch_size=1)
test_pcd_dataset_night = PCDDataset(test_data_night)
test_pcd_dataloader_night = DataLoader(test_pcd_dataset_night,
                        shuffle=False,
                        num_workers=0,
                        batch_size=1)
test_pcd_dataset_sunny = PCDDataset(test_data_sunny)
test_pcd_dataloader_sunny = DataLoader(test_pcd_dataset_sunny,
                        shuffle=False,
                        num_workers=0,
                        batch_size=1)