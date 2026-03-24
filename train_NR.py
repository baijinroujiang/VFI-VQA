
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import *
import VQA_model

from torchvision import transforms

import time

from scipy import stats
from scipy.optimize import curve_fit

import logging
# from torch.utils.tensorboard import SummaryWriter
from utils import performance_fit
from utils import L1RankLoss
import loss as ls

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    return y_output_logistic

def set_logging(config):
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )
  
def save_model(config, model, old_save_name, epoch, performance):
    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    if os.path.exists(old_save_name):
        os.remove(old_save_name)
    save_model_name = os.path.join(config.ckpt_path,
                                   config.model_name + '_' + config.database + '_NR_v' + str(
                                       config.exp_version) + '_epoch_%d_SRCC_%f.pth' % (
                                       epoch + 1, performance))
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), save_model_name)
    else:
        torch.save(model.state_dict(), save_model_name)
    return save_model_name

def main(config):
    # set_logging(config)
    # logging.info(config)
    # writer = SummaryWriter(os.path.join(config.log_path, config.log_file[:-4]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = eval('VQA_model.{}()'.format(config.model_name))

    if config.load_name is not None:
        para_in_last_net = torch.load(config.load_name)
        model.load_state_dict(para_in_last_net)

    if config.multi_gpu:
        model = torch.nn.DataParallel(model)
        model = model.to(device)
    else:
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr = config.conv_base_lr, weight_decay = config.weight_decay)

    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)
    if config.loss_type == 'MSE':
        criterion = nn.MSELoss().to(device)
        print('MSE LOSS')
    elif config.loss_type == 'L1RankLoss':
        criterion = L1RankLoss()
        print('L1Rank LOSS')

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))
    videos_dir = config.videos_dir
    datainfo = config.datainfo

    transformations_train = transforms.Compose(
        [transforms.Resize([config.imgsize, config.imgsize]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    transformations_test = transforms.Compose(
        [transforms.Resize([config.imgsize, config.imgsize]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    trainset = BVIDataset(videos_dir, datainfo, transformations_train, is_train=True)
    valset = BVIDataset(videos_dir, datainfo, transformations_test, is_train=False)
    testset = BVIDataset(videos_dir, datainfo, transformations_test, is_train=False, is_test=True)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
        shuffle=True, num_workers=config.num_workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)
    
    best_criterion = -1 
    best_result = []
    old_save_name = 'None'
    best_result = [-1,-1,-1,-1]

    for epoch in range(config.epochs):
        model.train()
        batch_losses = []
        batch_losses_each_disp = []

        for i, (video_ref, video_dis, dmos, _) in enumerate(train_loader):
          
            video_ref = video_ref.to(device)
            video_dis = video_dis.to(device)
            labels = dmos.to(device).float()

            outputs = model(video_ref, video_dis)

            optimizer.zero_grad()
            loss = criterion(labels, outputs)
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())
            loss.backward()

            optimizer.step()

            if (i + 1) % (config.print_samples // config.train_batch_size) == 0:
                avg_loss_epoch = sum(batch_losses_each_disp) / (config.print_samples // config.train_batch_size)
                batch_losses_each_disp = []

                print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' % (
                    epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size, avg_loss_epoch))

                # logging.info(outputs.cpu().detach().numpy())
                # logging.info(labels.cpu().detach().numpy())


        avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size)
        print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

        scheduler.step()
        lr = scheduler.get_last_lr()
        print('The current learning rate is {:.06f}'.format(lr[0]))

        with torch.no_grad():
            model.eval()
            label_val = np.zeros([len(valset)])
            val_output = np.zeros([len(valset)])
            for i, (video_ref, video_dis, dmos, _) in enumerate(val_loader):

                video_ref = video_ref.to(device)
                video_dis = video_dis.to(device)
                outputs = model(video_ref, video_dis)

                label_val[i] = dmos.item()
                val_output[i] = outputs.item()

            # logging.info(label_val[:5])
            # logging.info(val_output[:5])
            val_loss = criterion(torch.FloatTensor(label_val), torch.FloatTensor(val_output))
            print('Val loss:{:.4f}'.format(val_loss.item()))

            val_SRCC, val_KRCC, val_PLCC, val_RMSE = performance_fit(label_val, val_output)

            print(
                'Epoch {} completed. The result on the val databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, '
                'and RMSE: {:.4f}'.format(epoch + 1, val_SRCC, val_KRCC, val_PLCC, val_RMSE*100))

            # if epoch == 0:
            #     logging.info(label)
            # else:
            #     logging.info(label[:5])
            # if epoch % 5 == 0:
            #     plot_and_save(label, y_output, epoch, config)

            selected_performance = [val_SRCC, val_KRCC, val_PLCC, val_RMSE * 100]
            if selected_performance[0] > best_criterion:
                print("Update best model using best_criterion in epoch {}".format(epoch + 1))
                best_criterion = selected_performance[0]
                old_save_name = save_model(config, model, old_save_name, epoch, selected_performance[0])
                # plot_and_save(label, y_output, epoch, config, 'updata')
      
        with torch.no_grad():
              model.eval()
              label = np.zeros([len(testset)])
              y_output = np.zeros([len(testset)])
              for i, (video_ref, video_dis, dmos, _) in enumerate(test_loader):
  
                  video_ref = video_ref.to(device)
                  video_dis = video_dis.to(device)
                  outputs = model(video_ref, video_dis)
  
                  label[i] = dmos.item()
                  y_output[i] = outputs.item()
  
              test_SRCC, test_KRCC, test_PLCC, test_RMSE = performance_fit(label, y_output)
              print(
                  'The result on the test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                      test_SRCC, test_KRCC, test_PLCC, test_RMSE*100))


    print('Training completed.')
    print(
        'The best training result on the dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
            best_result[0], best_result[1], best_result[2], best_result[3]))
    if not os.path.exists(config.results_path):
        os.makedirs(config.results_path)
    np.save(
        os.path.join(config.results_path, config.model_name + '_' + config.database + '_v' + str(config.exp_version)),
        best_result)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--database', type=str, default='BVIVFI')
    parser.add_argument('--model_name', type=str, default='Res3d18_NR')

    parser.add_argument('--conv_base_lr', type=float, default=0.0001)
    parser.add_argument('--datainfo', type=str, default='./DATASET/json_files/vfi/train_BVI_000.json')
    parser.add_argument('--videos_dir', type=str, default='./DATASET/BVI-VFI/frames/')
    parser.add_argument('--decay_ratio', type=float, default=0.8)
    parser.add_argument('--decay_interval', type=int, default=50)
    parser.add_argument('--results_path', type=str, default='./result')
    parser.add_argument('--exp_version', type=int, default=1)
    parser.add_argument('--print_samples', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    
    # parser.add_argument('--log_path', type=str, default="./output/")
    # parser.add_argument('--log_file', type=str, default="Res3d18_NR.txt")
    parser.add_argument('--load_name', type=str, default="/home/hanjinliang/DATA2/VL/InternVL-main/DATA/extract_features/Res3d18d2cm_BVIVFIV3_FR_v0_epoch_73_SRCC_0.841594.pth")

    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--imgsize', type=int, default=256)

    parser.add_argument('--loss_type', type=str, default='MSE')

    config = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    main(config)
