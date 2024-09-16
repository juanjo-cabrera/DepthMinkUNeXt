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
from eval.evaluation_utils import FreiburgPCDMap, test_pcd_dataloader_cloudy, test_pcd_dataloader_night, test_pcd_dataloader_sunny, map_data
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
            batch, test_coor = data[0], data[1].to(device)
            #points = test_points['points']
            #color = test_points['colors']

            #points = points.reshape(points.shape[1], points.shape[2])
            #color = color.reshape(color.shape[1], color.shape[2])
            #coords, feats = ME.utils.sparse_quantize(coordinates=points, features=color, quantization_size=PARAMS.quantization_size)

            #bcoords = ME.utils.batched_coordinates([coords])
            #if not PARAMS.use_rgb:
            #    feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
            #feats = torch.ones((bcoords.shape[0], 1), dtype=torch.float32).cuda()
            #batch = {'coords': bcoords.to(device), 'features': feats.to(device)}
            test_descriptor = model(batch)['global'].detach().cpu() #.numpy()

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

    results = PARAMS.dataset_folder + 'results/Minkunext_exp4_v3.csv'

    with open(results, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Test Dataset", "MAE (m) ", 'Varianza (m2)', 'Desv. (m)', 'MSE (m)',
             'RMSE (m)', 'Mean Time (s)'])

        # Carga de forma secuencial todos los modelos que encuentres en la carpeta PARAMS.dataset_folder + 'models/Exp1_60epochs/'
        # los modelos tienen que acabar en .pth
        #models_names = sorted(os.listdir(PARAMS.dataset_folder + 'models/' + PARAMS.experiment_name + '/'))
        #models_names = [model_name for model_name in models_names if model_name.endswith('.pth')]
        PARAMS.use_rgb = True
        if PARAMS.use_rgb:
                        model.conv0p1s1 = ME.MinkowskiConvolution(
                            3, 32, kernel_size=5, dimension=3)
        best_model_path = '/home/arvc/Juanjo/develop/IndoorMinkUnext/weights/Indoor_MinkUNeXt_RGBpos0.6neg0.6voxel_size0.05height-0.25_20240904_1036_best.pth'        
        final_model_path = '/home/arvc/Juanjo/develop/IndoorMinkUnext/weights/Indoor_MinkUNeXt_RGBpos0.6neg0.6voxel_size0.05height-0.25_20240904_1036_final.pth'
        best_model_name = 'Indoor_MinkUNeXt_RGBpos0.6neg0.6voxel_size0.05height-0.25_20240904_1036_best'
        final_model_name = 'Indoor_MinkUNeXt_RGBpos0.6neg0.6voxel_size0.05height-0.25_20240904_1036_final'
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)
        print('Best model loaded from: {}'.format(best_model_path))


        mae_cloudy, varianza_cloudy, desv_cloudy, mse_cloudy, rmse_cloudy, mean_processing_time_cloudy = test(model, map_data,
                                                                                test_pcd_dataloader_cloudy)

        mae_night, varianza_night, desv_night, mse_night, rmse_night, mean_processing_time_night = test(model, map_data,
                                                                                test_pcd_dataloader_night)
        mae_sunny, varianza_sunny, desv_sunny, mse_sunny, rmse_sunny, mean_processing_time_sunny = test(model, map_data,
                                                                                test_pcd_dataloader_sunny)
        
        
        # write results to a .txt withou deleting previous results
        file_name = '/home/arvc/Juanjo/develop/IndoorMinkUnext/training/experiment4_results_v3.txt'
        mean_mae = (mae_cloudy + mae_night + mae_sunny) / 3
        with open(file_name, "a") as f:
            # write header
            if os.stat(file_name).st_size == 0:
                f.write('Model, MAE_Cloudy, MAE_Night, MAE_Sunny, Mean_MAE, Varianza_Cloudy, Varianza_Night, Varianza_Sunny, Desv_Cloudy, Desv_Night, Desv_Sunny, MSE_Cloudy, MSE_Night, MSE_Sunny, RMSE_Cloudy, RMSE_Night, RMSE_Sunny, Mean_Processing_Time_Cloudy, Mean_Processing_Time_Night, Mean_Processing_Time_Sunny\n')
                
            f.write(f'{best_model_name}, {mae_cloudy}, {mae_night}, {mae_sunny}, {mean_mae} {varianza_cloudy}, {varianza_night}, {varianza_sunny}, {desv_cloudy}, {desv_night}, {desv_sunny}, {mse_cloudy}, {mse_night}, {mse_sunny}, {rmse_cloudy}, {rmse_night}, {rmse_sunny}, {mean_processing_time_cloudy}, {mean_processing_time_night}, {mean_processing_time_sunny}\n')
            print('Results saved to: ', file_name)

        # Evaluate the final model too
        model.load_state_dict(torch.load(final_model_path))
        model.to(device)
        print('Final model loaded from: {}'.format(final_model_path))
        mae_cloudy, varianza_cloudy, desv_cloudy, mse_cloudy, rmse_cloudy, mean_processing_time_cloudy = test(model, map_data,
                                                                                test_pcd_dataloader_cloudy)
        
        mae_night, varianza_night, desv_night, mse_night, rmse_night, mean_processing_time_night = test(model, map_data,
                                                                                test_pcd_dataloader_night)
        mae_sunny, varianza_sunny, desv_sunny, mse_sunny, rmse_sunny, mean_processing_time_sunny = test(model, map_data,
                                                                                    test_pcd_dataloader_sunny)
        
        # write results to a .txt withou deleting previous results
        mean_mae = (mae_cloudy + mae_night + mae_sunny) / 3
        with open(file_name, "a") as f:
            f.write(f'{final_model_name}, {mae_cloudy}, {mae_night}, {mae_sunny}, {mean_mae} {varianza_cloudy}, {varianza_night}, {varianza_sunny}, {desv_cloudy}, {desv_night}, {desv_sunny}, {mse_cloudy}, {mse_night}, {mse_sunny}, {rmse_cloudy}, {rmse_night}, {rmse_sunny}, {mean_processing_time_cloudy}, {mean_processing_time_night}, {mean_processing_time_sunny}\n')
            print('Results saved to: ', file_name)



