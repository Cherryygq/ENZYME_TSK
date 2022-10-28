"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from ModelNetDataLoader import ModelNetDataLoader
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_cls_ssg', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=6, type=int, help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0005, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1000, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default='pointnet2_cls_ssg', help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=True, help='use normals')##是否使用特征
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')##是否使用均匀采样
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(criterion, model, loader, num_class=6):
    mean_correct = []
    mean_loss = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, fea = classifier(points)
        pred_choice = pred.data.max(1)[1]

        # add
        loss = criterion(pred, target.long(), fea)
        mean_loss.append(loss.cpu().detach().numpy())

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    instance_loss = np.mean(mean_loss)

    return instance_acc, class_acc, instance_loss



def plot_losses(epoch,train_acc,train_loss,val_acc,val_loss):
    plt.figure()
    plt.plot(range(epoch), train_loss)
    plt.plot(range(epoch), val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('model_loss.png', bbox_inches='tight')

    plt.figure()
    plt.plot(range(epoch), train_acc)
    plt.plot(range(epoch), val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('model_accuracy.png', bbox_inches='tight')



def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(args.gpu)

    '''CREATE DIR'''                             ##创建log目录
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    print(timestr)
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)##exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常。

    '''LOG'''                                           ##日志
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = './data/enzyme_data/'

    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)##导入一个模块
    shutil.copy('./%s.py' % args.model, str(exp_dir))##shutil.copyfile(src, dst)：复制文件内容（不包含元数据）从src到dst。
    shutil.copy('pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)##args.use_normals=false：不使用点的其他特征
    criterion = model.get_loss()
    classifier.apply(inplace_relu)
    # print(classifier.summary())

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()
    print(classifier)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        mean_loss = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()


            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)

            pred_choice = pred.data.max(1)[1]
            # print("pred_choice",pred_choice)
            # print("target.long().data", target.long().data)
            correct = pred_choice.eq(target.long().data).cpu().sum()

            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1
            mean_loss.append(loss.cpu().detach().numpy())
        train_instance_acc = np.mean(mean_correct)
        train_instance_loss = np.mean(mean_loss)
        log_string('Train Instance Accuracy: %f,loss: %f' % (train_instance_acc, train_instance_loss))


        with torch.no_grad():
            instance_acc, class_acc, val_loss = test(criterion, classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f, test loss: %f' % (instance_acc, class_acc, val_loss))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1
        train_acc_list.append(train_instance_acc)
        train_loss_list.append(train_instance_loss)
        val_acc_list.append(instance_acc)
        val_loss_list.append(val_loss)
    # print(train_loss_list)
    # print(len(train_loss_list))
    # print(train_acc_list)
    # print(len(train_acc_list))
    # print(len(val_acc_list))
    # print(len(val_loss_list))
    plot_losses(args.epoch,train_acc_list,train_loss_list,val_acc_list,val_loss_list)
    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
