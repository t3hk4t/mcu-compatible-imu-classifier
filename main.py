from cProfile import run
import copy
import json
import math
from multiprocessing import reduction
import os
import torch.onnx
import torchmetrics
import numpy as np
import logging
from importlib import import_module
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from torchmetrics import ConfusionMatrix
from ray import tune
from functools import partial
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers import ASHAScheduler

from datalaoder.LinearDataLoader import Dataset
from preprocessor.preprocess import PreProcessor
from modules.file_utils import FileUtils
from modules import tensorboard_utils
from modules.loss_functions import *
from modules.csv_utils_2 import CsvUtils2

plt.style.use('dark_background')
plt.ion()

enabled_axis = [
        [1, 1, 1, 1, 1, 1],
        [0, 1, 1, 0, 1, 1],
        [0, 0, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ]


def load_data(config):
    dataset = Dataset(config['dataset_path'], config['enabled_axis'], config['overlap'],
                      config['buffer_size'], config['sensor_axis'],
                      normalize=config['normalize_dataset'])
    labels = dataset.get_labels()
    feature_lengths = dataset.get_feature_lengths()
    train_test_split = int(len(dataset) * config['train_test_split'])
    dataset_train, dataset_test = torch.utils.data.random_split(
        dataset,
        [train_test_split, len(dataset) - train_test_split],
        generator=torch.Generator().manual_seed(0)
    )
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=config['batch_size'],
        shuffle=True
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=config['batch_size'],
        shuffle=False
    )

    return dataloader_train, dataloader_test, labels, feature_lengths


def train_classifier(config):
    config['enabled_axis'] = enabled_axis[config['enabled_axis']]
    temp = config['sequence_name']
    path_sequence = f'.{os.sep}results{os.sep}{temp}'
    config['run_name'] += ('-' + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))
    run_name = config['run_name']
    path_run = f'.{os.sep}results{os.sep}{temp}{os.sep}{run_name}'
    FileUtils.createDir(path_run)
    USE_CUDA = config['is_cuda']
    CsvUtils2.create_global(path_sequence)
    CsvUtils2.create_local(path_sequence, config['run_name'])
    
    log_formatter = logging.Formatter("%(asctime)s [%(process)d] [%(thread)d] [%(levelname)s]  %(message)s")
    base_name = os.path.basename(path_sequence)
    file_handler = logging.FileHandler(f'{path_run}/log-{base_name}.txt')
    file_handler.setFormatter(log_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    tensorboard_writer = tensorboard_utils.CustomSummaryWriter(log_dir=path_run)
    tensorboard_helper = tensorboard_utils.TensorBoardUtils(tensorboard_writer)

    device = torch.device('cuda:0' if False else 'cpu')

    dataloader_train, dataloader_test, labels, feature_lengths = load_data(config)

    Model = getattr(__import__('models.' + config['model'], fromlist=['Model']), 'Model')
    model = Model(config, len([i for i, x in enumerate(config['enabled_axis']) if x]),
                  config['conv1_out_channels'], config['conv2_out_channels']).to(device)
    param_cnt, model_size = model.calculate_model_stats()
    config['model_size'] = model_size
    config['learnable_param_count'] = param_cnt

    LossFunction = getattr(import_module(f'torch.nn'), config['loss_function'])
    if config['use_feature_weights']:
        if config['weight_calculation_type'] == 1:
            feature_weights = 1 - feature_lengths/np.sum(feature_lengths)
        else:
            feature_weights = (1 / feature_lengths) * np.sum(feature_lengths)
        loss_fn = LossFunction(weight=torch.tensor(feature_weights).float(), reduction='mean').to(device)
    else:
        loss_fn = LossFunction().to(device)


    Optimizer = getattr(import_module(f'torch.optim'), config['optimizer'])
    optimizer = Optimizer(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    state_dict = {
        'train_loss': -1.0,
        'test_loss': -1.0,
        'train_accuracy': -1.0,
        'test_accuracy': -1.0,
        'best_accuracy': -1.0,
        'early_stopping_patience': 0.0,
        'early_percent_improvement': 0.0,
    }

    metric_accuracy_train = torchmetrics.Accuracy()
    metric_accuracy_test = torchmetrics.Accuracy()
    metric_confusion_matrix = ConfusionMatrix(num_classes=4)
    train_loss = []
    test_loss = []
    best_accuracy = 0.0

    for epoch in range(1, config['epochs']):
        state_before = copy.deepcopy(state_dict)
        for dataloader in [dataloader_train, dataloader_test]:
            losses = []

            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                y_prim = model.forward(x.float())
                loss = loss_fn.forward(y_prim.float(), y)
                losses.append(loss.cpu().item())

                if dataloader == dataloader_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    metric_accuracy_train(y_prim.cpu(), y.cpu())
                else:
                    metric_confusion_matrix(y_prim.cpu(), y.cpu())
                    metric_accuracy_test(y_prim.cpu(), y.cpu())

            if dataloader == dataloader_train:
                train_loss.append(np.mean(losses))
            else:
                test_loss.append(np.mean(losses))

            losses.clear()

        early_stopping = state_dict['early_stopping_patience']

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(mean_accuracy=float(metric_accuracy_test.compute()), test_loss = test_loss[-1], model_size = model_size)

        state_dict['train_loss'] = train_loss[-1]
        state_dict['test_loss'] = test_loss[-1]
        state_dict['train_accuracy'] = metric_accuracy_train.compute()
        state_dict['test_accuracy'] = metric_accuracy_test.compute()
        state_dict['best_accuracy'] = best_accuracy

        if metric_accuracy_test.compute() > best_accuracy:
            best_accuracy = metric_accuracy_test.compute()
            torch.save(model.state_dict(), f'best_loss.pt')
            model.eval()
            torch.onnx.export(model, torch.randn(1, len([i for i, x in enumerate(config['enabled_axis']) if x]), 200, device="cpu"),
                              f'best_loss.onnx', verbose=False,
                              input_names=['sensor_data'], output_names=['predicted_state'],
                              opset_version=10,
                              export_params=True,
                              do_constant_folding=True,
                              dynamic_axes={
                                  'input': {0: 'batch_size'},
                                  'output': {0: 'batch_size'}
                              })
            model.train()

        percent_improvement = 0
        if epoch > 1:
            if state_before[config['early_stopping_param']] != 0:
                percent_improvement = (
                                              state_dict[config['early_stopping_param']] - state_before[
                                          config['early_stopping_param']]) / \
                                      state_before[config['early_stopping_param']]
            if state_dict[config['early_stopping_param']] >= 0:
                if config['early_stopping_delta_percent'] > percent_improvement:
                    state_dict['early_stopping_patience'] += 1.0
                else:
                    state_dict['early_stopping_patience'] = 0.0
            state_dict['early_percent_improvement'] = percent_improvement

        if state_dict['early_stopping_patience'] >= config['early_stopping_patience'] or \
                math.isnan(percent_improvement):
            break

        tensorboard_helper.addPlotConfusionMatrix(metric_confusion_matrix.compute().numpy(), labels,
                                                  'Confusion Matrix', global_step=epoch)

        tensorboard_writer.add_hparams(
            hparam_dict=config,
            global_step=epoch,
            name=config['run_name'],
            metric_dict=state_dict
        )

        CsvUtils2.add_hparams(
            path_sequence=path_sequence,
            run_name=config['run_name'],
            args_dict=config,
            metrics_dict=state_dict,
            global_step=epoch
        )

        metric_confusion_matrix.reset()
        metric_accuracy_test.reset()
        metric_accuracy_train.reset()
        tensorboard_writer.flush()


def main():
    parser = argparse.ArgumentParser(description='Model trainer')
    parser.add_argument('-run_name', default=f'', type=str)
    parser.add_argument('-sequence_name', default=f'robot_state_classification', type=str)
    parser.add_argument('-checkpoint_dir', default=f'/tmp', type=str)
    parser.add_argument('-result_output_dir', default=f'./', type=str)
    parser.add_argument('-clear_logs', default=False, type=bool)
    parser.add_argument('-dataset_path', default=r'/home/leonards/Documents/robot_tsc/data_custom_ml/dataset.hdf5', type=str)
    parser.add_argument('-normalize_dataset', default=False, type=bool)
    parser.add_argument('-overlap', default=100, type=int)
    parser.add_argument('-train_test_split', default=0.7, type=float)
    parser.add_argument('-learning_rate', default=1e-4, type=float)
    parser.add_argument('-momentum', default=0.95, type=float)
    parser.add_argument('-weight_decay', default=2e-5, type=float)
    parser.add_argument('-batch_size', default=64, type=int)
    parser.add_argument('-model', default='cnn', type=str)  
    parser.add_argument('-use_feature_weights', default=False, type=bool)  
    parser.add_argument('-weight_calculation_type', default=1, type=int)   # 1 or 2
    parser.add_argument('-sensor_axis', default=6, type=int)
    parser.add_argument('-preprocess_data', default=True, type=bool)
    parser.add_argument('-enabled_axis', default=0, type=int)
    parser.add_argument('-output_size', default=4, type=int)
    parser.add_argument('-grid_search', default=True, type=bool)
    parser.add_argument('-loss_function', default='CrossEntropyLoss', type=str)
    parser.add_argument('-optimizer', default='RAdam', type=str)
    parser.add_argument('-epochs', default=400, type=int)
    parser.add_argument('-debug', default=True, type=bool)
    parser.add_argument('-buffer_size', default=200, type=int)
    parser.add_argument('-conv1_out_channels', default=32, type=int)
    parser.add_argument('-conv2_out_channels', default=16, type=int)
    parser.add_argument('-is_cuda', default=torch.cuda.is_available(), type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-continue_training', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-learnable_param_count', default=4024, type=int)
    parser.add_argument('-model_size', default=20.0, type=float)
    parser.add_argument('-early_stopping_patience', default=50, type=int)
    parser.add_argument('-early_stopping_param', default='best_accuracy', type=str)
    parser.add_argument('-early_stopping_delta_percent', default=0.00005, type=float)
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = main()
    config = args.__dict__

    if config['preprocess_data']:
        print('Preprocessing data')
        preprocessor = PreProcessor()
        preprocessor.process_data()
        print('Finished preprocessing data')


    if config['grid_search']:
        config['learning_rate'] = tune.loguniform(1e-5, 1e-2)
        config['conv2_out_channels'] = tune.choice([4, 8, 12, 16, 20])
        config['conv1_out_channels'] = tune.choice([8, 16, 24, 32])
        config['batch_size'] = tune.choice([16, 32, 64, 128, 256])
        config['optimizer'] = tune.choice(['RAdam', 'Adam'])
        config['weight_decay'] = tune.loguniform(1e-5, 1e-3)
        config['enabled_axis'] = tune.choice([0, 1, 2, 3, 4, 5, 6])

        scheduler = ASHAScheduler(metric="mean_accuracy", mode="max", grace_period=120, max_t=config['epochs'])

        result = tune.run(
            partial(train_classifier),
            name="grid_search_classification",
            resources_per_trial={
                "cpu": 1,
                "gpu" : False
                },
                config=config,
            num_samples=100,
            scheduler=scheduler,
            keep_checkpoints_num=1,
            checkpoint_score_attr="mean_accuracy",
            sync_config=tune.SyncConfig(
                syncer=None
            ),
            local_dir=config['checkpoint_dir']) 

        best_trial = result.get_best_trial("mean_accuracy", "max", "last-5-avg")
        best_logdir = result.get_best_logdir("mean_accuracy", "max", "last-5-avg")

        print("Best trial config: {}".format(best_trial.config))
        print("Best trial path: {}".format(best_logdir))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["test_loss"]))
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["mean_accuracy"]))

        best_trial_enabled_axis_idx = best_trial.config['enabled_axis']
        best_trial_enabled_axis = enabled_axis[best_trial_enabled_axis_idx]
        best_trial_enabled_axis_json = {
            'gyroscope_x' : bool(best_trial_enabled_axis[0]),
            'gyroscope_y' : bool(best_trial_enabled_axis[1]),
            'gyroscope_z' : bool(best_trial_enabled_axis[2]),
            'acc_x' : bool(best_trial_enabled_axis[3]),
            'acc_y' : bool(best_trial_enabled_axis[4]),
            'acc_z' : bool(best_trial_enabled_axis[5]),
        }

        print(f"----------Enabled axis----------")
        print(f"gyroscope_x = {best_trial_enabled_axis_json['gyroscope_x']}")
        print(f"gyroscope_y = {best_trial_enabled_axis_json['gyroscope_y']}")
        print(f"gyroscope_z = {best_trial_enabled_axis_json['gyroscope_z']}")
        print(f"acc_x = {best_trial_enabled_axis_json['acc_x']}")
        print(f"acc_y = {best_trial_enabled_axis_json['acc_y']}")
        print(f"acc_z = {best_trial_enabled_axis_json['acc_z']}")
        print(f"----------Enabled axis----------")

        with open(f'{args.result_output_dir}{os.sep}enabled_axis.json', 'w') as fp:
            json.dump(best_trial_enabled_axis_json, fp)



        os.system(f"cp {best_logdir}{os.sep}best_loss.onnx {args.result_output_dir}")
        os.system(f"cp {best_logdir}{os.sep}best_loss.pt {args.result_output_dir}")
        os.system(f"cp {best_logdir}{os.sep}result.json {args.result_output_dir}")
        if config['clear_logs']:
            os.system(f"rm -r -f {os.sep}tmp{os.sep}grid_search_classification")

    else:
        train_classifier(config)
