import os
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config import PARAMS 
import numpy as np

import open3d as o3d
from datasets.base_datasets import PointCloudLoader
from scipy.spatial import cKDTree


class PNVPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Point clouds are already preprocessed with a ground plane removed
        self.remove_zero_points = False
        self.remove_ground_plane = False
        self.ground_plane_level = None

    

    def global_normalize(self, pcd, max_distance=15.0):
            import copy
            """
            Normalize a pointcloud to achieve mean zero, scaled between [-1, 1] and with a fixed number of points
            """
            pcd = copy.deepcopy(pcd)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

            [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            z_mean = np.mean(z)

            x = x - x_mean
            y = y - y_mean
            z = z - z_mean

            x = x / max_distance
            y = y / max_distance
            z = z / max_distance

            points[:, 0] = x
            points[:, 1] = y
            points[:, 2] = z

            # pointcloud_normalized = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            return points, colors
    
    def global_normalize_without_color(self, points, max_distance=15.0):
            import copy
            """
            Normalize a pointcloud to achieve mean zero, scaled between [-1, 1] and with a fixed number of points
            """

            [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            z_mean = np.mean(z)

            x = x - x_mean
            y = y - y_mean
            z = z - z_mean

            x = x / max_distance
            y = y / max_distance
            z = z / max_distance

            points[:, 0] = x
            points[:, 1] = y
            points[:, 2] = z

            # pointcloud_normalized = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            return points
    
    
    def filter_by_height(self, pcd=None, height=0.5):
        # filter the points by height
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        idx = points[:, 2] > height
        # now select the final pointclouds
        pcd_non_plane = o3d.geometry.PointCloud()
        pcd_non_plane.points = o3d.utility.Vector3dVector(points[idx])
        pcd_non_plane.colors = o3d.utility.Vector3dVector(colors[idx])

        # show pointcloud
        #o3d.visualization.draw_geometries([pcd_non_plane])
        
        return pcd_non_plane
    
    def filter_by_height_features(self, pcd, features, height=0.5):
        # filter the points by height
        points = np.asarray(pcd.points)
        idx = points[:, 2] > height

        return points[idx], features[idx]
    
    def read_pointcloud_with_features(self, path):
        """
        Lee una nube de puntos y sus características desde un archivo PLY.

        Args:
            path (str): Ruta del archivo PLY.

        Returns:
            pcd (open3d.geometry.PointCloud): Nube de puntos leída.
            features (np.ndarray): Array con las características asociadas a cada punto.
        """
        # Abrir el archivo PLY y leer el contenido
        with open(path, 'r') as ply_file:
            lines = ply_file.readlines()

        # Encontrar el header y la posición de los datos
        header_end_index = lines.index("end_header\n")
        header = lines[:header_end_index + 1]
        
        # Contar la cantidad de propiedades (coordenadas + features)
        properties = [line for line in header if line.startswith("property")]
        num_features = len(properties) - 3  # Restar las 3 coordenadas (x, y, z)

        # Leer los datos de los puntos
        point_data = np.loadtxt(lines[header_end_index + 1:])

        # Separar las coordenadas (x, y, z) y los features
        points = point_data[:, :3]  # Coordenadas (x, y, z)
        features = point_data[:, 3:]  # Features adicionales

        # Crear la nube de puntos con Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        return pcd, features
    def voxel_downsample_with_features(self, points, features, voxel_size):
        """
        Realiza un voxel downsampling en una nube de puntos y mantiene las características asociadas.
        
        Args:
            points (np.ndarray): Array de Nx3 con las coordenadas de los puntos.
            features (np.ndarray): Array de NxM con las características asociadas a cada punto.
            voxel_size (float): Tamaño del voxel para el downsampling.

        Returns:
            downsampled_points (np.ndarray): Puntos downsampled (Mx3).
            downsampled_features (np.ndarray): Features promediadas o combinadas correspondientes (MxM).
        """
        # Crear la nube de puntos en Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Realizar el voxel downsampling usando Open3D
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        downsampled_points = np.asarray(downsampled_pcd.points)

        # Crear un KD-Tree para asociar los puntos downsampled con los originales
        kdtree = cKDTree(points)

        # Buscar los índices de los puntos originales más cercanos a los puntos downsampled
        indices = kdtree.query_ball_point(downsampled_points, voxel_size)

        # Calcular las features combinadas (promedio) para los puntos en cada voxel
        downsampled_features = np.array([features[idx].mean(axis=0) for idx in indices if len(idx) > 0])

        return downsampled_points, downsampled_features

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        # Returns Nx3 ndarray
        file_path = os.path.join(file_pathname)
        if not PARAMS.use_dino_features:            
            pcd = o3d.io.read_point_cloud(file_path)
            # filter the points by height
            if PARAMS.height is not None:
                pcd = self.filter_by_height(pcd, height=PARAMS.height)       
            # show pointcloud
            #o3d.visualization.draw_geometries([pcd])
            if PARAMS.voxel_size is not None:
                pcd = pcd.voxel_down_sample(voxel_size=PARAMS.voxel_size)

            
            points, colors = self.global_normalize(pcd, max_distance=PARAMS.max_distance)
            pc = {}
            pc['points'] = points
            pc['colors'] = colors
        else:
            pcd, features = self.read_pointcloud_with_features(file_pathname)
    
            # filter the points by height
            if PARAMS.height is not None:
                points, features = self.filter_by_height_features(pcd, features, height=PARAMS.height)       
            # show pointcloud
            #o3d.visualization.draw_geometries([pcd])
            if PARAMS.voxel_size is not None:
                points, features = self.voxel_downsample_with_features(points, features, voxel_size=PARAMS.voxel_size)        
                points = self.global_normalize_without_color(points, max_distance=PARAMS.max_distance)
                pc = {}
                pc['points'] = points
                pc['colors'] = features
    
        return pc
