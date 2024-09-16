import sys
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from datasets import *
import csv
from config import PARAMS
import os
from eval.evaluation_utils_old import FreiburgPCDMap, test_pcd_dataloader_cloudy, test_pcd_dataloader_night, test_pcd_dataloader_sunny, map_data
import numpy as np
from model.minkunext import model
import MinkowskiEngine as ME
import torch


def compute_errors(errors):
    errors_cuadrado = np.power(errors, 2)
    mae = np.mean(errors)
    mse = np.mean(errors_cuadrado)
    rmse = np.sqrt(mse)
    varianza = np.mean(np.power(errors - mae, 2))
    desv = np.sqrt(varianza)
    return mae, varianza, desv, mse, rmse


def test(model, map_data, test_dataloader):
    device = torch.device(PARAMS.cuda_device if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = model.to(device)
    model.eval()


    freiburg_map = FreiburgPCDMap(map_data, model)

    # Testing Accuracy

    errors = []
    total = 0
    correct = 0
    times = []

    with torch.no_grad():
        for data in test_dataloader:
            test_points, test_coor = data[0], data[1].to(device)
            points = test_points['points']
            color = test_points['colors']

            points = points.numpy().reshape(points.shape[1], points.shape[2])
            color = color.numpy().reshape(color.shape[1], color.shape[2])
            coords, feats = ME.utils.sparse_quantize(coordinates=points, features=color, quantization_size=PARAMS.quantization_size)
            #coords = ME.utils.sparse_quantize(coordinates=test_points,
            #                                  quantization_size=0.01).cuda()

            bcoords = ME.utils.batched_coordinates([coords])
            if PARAMS.use_rgb:
                feats = torch.cat(feats, dim=0)
            else:
                feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
            #feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32).cuda()
            batch = {'coords': bcoords.to(device), 'features': feats.to(device)}
            test_descriptor = model(batch)['global'].detach().cpu().numpy()

            error,  processing_time = freiburg_map.evaluate_error_position(test_descriptor, test_coor)
            errors.append(error)
            times.append(processing_time)
            total += 1

    mae, varianza, desv, mse, rmse = compute_errors(errors)
    print('Mean Absolute Error in test images (m):', mae)
    print('Varianza:', varianza)
    print('Desviacion', desv)
    print('Mean Square Error (m2)', mse)
    print('Root Mean Square Error (m)', rmse)
    mean_processing_time = np.mean(times)
    return mae, varianza, desv, mse, rmse, mean_processing_time


if __name__ == "__main__":
    device = torch.device(PARAMS.cuda_device if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)


    if not os.path.exists(PARAMS.dataset_folder + 'results/'):
        os.makedirs(PARAMS.dataset_folder + 'results/')
        print(f"Carpeta '{PARAMS.dataset_folder + 'results/'}' creada.")
    else:
        print(f"Carpeta '{PARAMS.dataset_folder + 'results/'}' ya existe.")

    results = PARAMS.dataset_folder + 'results/Minkunext_exp4.csv'

    with open(results, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Test Dataset", "MAE (m) ", 'Varianza (m2)', 'Desv. (m)', 'MSE (m)',
             'RMSE (m)', 'Mean Time (s)'])

        # Carga de forma secuencial todos los modelos que encuentres en la carpeta PARAMS.dataset_folder + 'models/Exp1_60epochs/'
        # los modelos tienen que acabar en .pth
        #models_names = sorted(os.listdir(PARAMS.dataset_folder + 'models/' + PARAMS.experiment_name + '/'))
        #models_names = [model_name for model_name in models_names if model_name.endswith('.pth')]

        weights_path = '/home/arvc/Juanjo/develop/IndoorMinkUnext/weights/Indoor_MinkUNeXt_EXP4_20240822_1305_best.pth'
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        # print(test_model.avgpool)
        print(model)

        mae, varianza, desv, mse, rmse, mean_processing_time = test(model, map_data,
                                                                              test_pcd_dataloader_cloudy)
        writer.writerow(['cloudy', mae, varianza, desv, mse, rmse, mean_processing_time])
        mae, varianza, desv, mse, rmse, mean_processing_time = test(model, map_data,
                                                                              test_pcd_dataloader_night)
        writer.writerow(['night', mae, varianza, desv, mse, rmse, mean_processing_time])
        mae, varianza, desv, mse, rmse, mean_processing_time = test(model, map_data,
                                                                              test_pcd_dataloader_sunny)
        writer.writerow(['sunny', mae, varianza, desv, mse, rmse, mean_processing_time])