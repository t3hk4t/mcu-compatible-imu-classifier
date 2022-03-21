import os
import torch.onnx
import torchmetrics
import numpy as np
import logging
from importlib import import_module
import matplotlib.pyplot as plt
import argparse
import time
from datetime import datetime
from torchmetrics import ConfusionMatrix
from PIL import Image

from datalaoder.LinearDataLoader import Dataset
from modules.file_utils import FileUtils
from modules import tensorboard_utils
from modules.loss_functions import *
from modules.csv_utils_2 import CsvUtils2

plt.style.use('dark_background')
plt.ion()


def main():
    parser = argparse.ArgumentParser(description='Model trainer')
    parser.add_argument('-run_name', default=f'run_{time.time()}', type=str)
    parser.add_argument('-sequence_name', default=f'robot_state_classification', type=str)
    parser.add_argument('-dataset_path', default=f'.{os.sep}data_custom_ml{os.sep}test_dataset.hdf5', type=str)
    parser.add_argument('-normalize_dataset', default=False, type=bool)
    parser.add_argument('-train_test_split', default=0.7, type=float)
    parser.add_argument('-learning_rate', default=1e-4, type=float)
    parser.add_argument('-momentum', default=0.95, type=float)
    parser.add_argument('-weight_decay', default=2e-5, type=float)
    parser.add_argument('-batch_size', default=64, type=int)
    parser.add_argument('-model', default='cnn', type=str)
    parser.add_argument('-sensor_axis', default=6, type=int)
    parser.add_argument('-output_size', default=4, type=int)
    parser.add_argument('-loss_function', default='CrossEntropyLoss', type=str)
    parser.add_argument('-optimizer', default='RAdam', type=str)
    parser.add_argument('-epochs', default=10000, type=int)
    parser.add_argument('-debug', default=True, type=bool)
    parser.add_argument('-buffer_size', default=200, type=int)
    parser.add_argument('-is_cuda', default=torch.cuda.is_available(), type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-continue_training', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-early_stopping_patience', default=10, type=int)
    parser.add_argument('-early_stopping_param', default='test_loss', type=str)
    parser.add_argument('-early_stopping_delta_percent', default=0.005, type=float)
    args, _ = parser.parse_known_args()
    # setting up some logging stuff
    path_sequence = f'.{os.sep}results{os.sep}{args.sequence_name}'
    args.run_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))
    path_run = f'.{os.sep}results{os.sep}{args.sequence_name}{os.sep}{args.run_name}'
    FileUtils.createDir(path_run)
    FileUtils.writeJSON(f'{path_run}/args.json', vars(args))
    USE_CUDA = args.is_cuda
    CsvUtils2.create_global(path_sequence)
    CsvUtils2.create_local(path_sequence, args.run_name)

    root_logger = logging.getLogger()
    log_formatter = logging.Formatter("%(asctime)s [%(process)d] [%(thread)d] [%(levelname)s]  %(message)s")
    root_logger.level = logging.DEBUG  # level
    base_name = os.path.basename(path_sequence)
    file_handler = logging.FileHandler(f'{path_run}/log-{base_name}.txt')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    tensorboard_writer = tensorboard_utils.CustomSummaryWriter(log_dir=path_run)
    tensorboard_helper = tensorboard_utils.TensorBoardUtils(tensorboard_writer)


    device = torch.device('cuda:0' if False else 'cpu')

    action_buffer = ['drive_off', 'drive_on', 'stand_off', 'stand_on']

    dataset = Dataset(args.dataset_path, normalize=args.normalize_dataset)
    train_test_split = int(len(dataset) * args.train_test_split)
    dataset_train, dataset_test = torch.utils.data.random_split(
        dataset,
        [train_test_split, len(dataset) - train_test_split],
        generator=torch.Generator().manual_seed(0)
    )
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=True
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        shuffle=False
    )

    Model = getattr(__import__('models.' + args.model, fromlist=['Model']), 'Model')
    model = Model(args).to(device)

    LossFunction = getattr(import_module(f'torch.nn'), args.loss_function)
    loss_fn = LossFunction().to(device)

    Optimizer = getattr(import_module(f'torch.optim'), args.optimizer)
    optimizer = Optimizer(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    hP = args.__dict__
    hP['dataset_path'] = ''

    state_dict = {
        'train_loss': -1.0,
        'test_loss': -1.0,
        'train_accuracy': -1.0,
        'test_accuracy': -1.0,
        'best_accuracy': -1.0,
    }

    metric_accuracy_train = torchmetrics.Accuracy()
    metric_accuracy_test = torchmetrics.Accuracy()
    metric_confusion_matrix = ConfusionMatrix(num_classes=4)
    train_loss = []
    test_loss = []
    best_accuracy = 0.0

    for epoch in range(1, args.epochs):
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

        root_logger.info(
            f'Epoch - {epoch} ------ Train loss : {train_loss[-1]}, Test loss : {test_loss[-1]}, Train accuracy : '
            f'{metric_accuracy_train.compute()},'
            f'Test accuracy : {metric_accuracy_test.compute()}')

        state_dict['train_loss'] = train_loss[-1]
        state_dict['test_loss'] = test_loss[-1]
        state_dict['train_accuracy'] = metric_accuracy_train.compute()
        state_dict['test_accuracy'] = metric_accuracy_test.compute()
        state_dict['best_accuracy'] = best_accuracy

        if metric_accuracy_test.compute() > best_accuracy:
            best_accuracy = metric_accuracy_test.compute()
            torch.save(model.state_dict(), './best_loss.pt')
            model.eval()
            torch.onnx.export(model, torch.randn(1, 6, 200, device="cpu"), './best_loss.onnx', verbose=False,
                              input_names=['sensor_data'], output_names=['predicted_state'],
                              opset_version=10,
                              export_params=True,
                              do_constant_folding=True,
                              dynamic_axes={
                                  'input': {0: 'batch_size'},
                                  'output': {0: 'batch_size'}
                              })
            model. train()

        tensorboard_helper.addPlotConfusionMatrix(metric_confusion_matrix.compute().numpy(), action_buffer, 'Confusion Matrix', global_step=epoch)

        tensorboard_writer.add_hparams(
            hparam_dict=hP,
            global_step=epoch,
            name=args.run_name,
            metric_dict=state_dict
        )

        CsvUtils2.add_hparams(
            path_sequence=path_sequence,
            run_name=args.run_name,
            args_dict=hP,
            metrics_dict=state_dict,
            global_step=epoch
        )

        metric_confusion_matrix.reset()
        metric_accuracy_test.reset()
        metric_accuracy_train.reset()

        tensorboard_writer.flush()


if __name__ == '__main__':
    main()
