import os
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'
import sys
import numpy as np
import yaml
import torch
from torch.nn.parallel import DistributedDataParallel

torch.set_float32_matmul_precision('high')
from torch import nn
import clip
import random
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from utils import *
from networks import *
from datasets import *


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, optimizer, train_loader, valid_loader, scheduler, log):
    max_performance = 0
    global_step = 0
    criterion = nn.CrossEntropyLoss()
    # scaler = torch.cuda.amp.GradScaler()
    save_dir = args.model_dir
    start_epoch = 0
    # checkpoint
    if args.resume > -1:
        print('Load Checkpoint From models_params_{}'.format(args.resume))
        checkpoint = torch.load(os.path.join(save_dir, 'models_params_{}.pth'.format(args.resume)),
                                map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict((checkpoint['optimizer_state_dict']))
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['step']
        scheduler.last_epoch = global_step
        scheduler.step(global_step)
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        total_num = 0
        correct_num = 0
        model.train()
        with torch.enable_grad():
            for i, data in enumerate(tqdm(train_loader)):
                for k, v in data.items():
                    if isinstance(v, list):
                        data[k] = v
                    elif data[k] is not None:
                        data[k] = v.to(device)
                labels = data['labels']
                optimizer.zero_grad()
                outputs, moe_loss = model(data)
                ce_loss = criterion(outputs, labels)
                loss = ce_loss + moe_loss
                loss.backward()
                optimizer.step()
                if args.rank == 0:
                    epoch_loss += loss.item()
                    total_num += outputs.size(0)
                    correct_num += torch.sum(torch.argmax(outputs, 1) == labels).item()
                    global_step += args.batch_size
                if global_step % args.record_step == 0 and args.rank == 0:
                    log.info(
                        'Epoch [{:0>3}/{:0>3}], step:{} lr:{} loss:{:.4f}'
                        .format(epoch + 1, args.epochs, global_step, scheduler.get_last_lr(), epoch_loss / total_num), )
                    epoch_loss = 0
                    total_num = 0

                if global_step % args.test_step == 0 and args.rank == 0:
                    log.info('Start Evaluation...: {}'.format(global_step))
                    model.eval()
                    state = {'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'epoch': epoch,
                             'step': global_step
                             }
                    torch.save(state, os.path.join(save_dir, 'models_params_{}.pth'.format(global_step)))

                    with torch.no_grad():
                        keys = valid_loader.keys()
                        log.info('Valid Loader Datasets:{}'.format(keys))
                        avg_auc = []
                        avg_acc = []
                        avg_eer = []
                        avg_vauc = []
                        for key in keys:
                            video_predictions = []
                            video_labels = []
                            frame_predictions = []
                            frame_labels = []
                            for data in tqdm(valid_loader[key]):
                                for k, v in data.items():
                                    if isinstance(v, list):
                                        data[k] = v
                                    elif data[k] is not None:
                                        data[k] = v.to(device)
                                labels = torch.where(data['labels'] < 1, 0, 1)
                                outputs,_ = model(data)
                                outputs = F.softmax(outputs, dim=-1)
                                # fake的概率
                                frame_predictions.extend(
                                    np.ones_like(outputs[:, 0].cpu()) - outputs[:, 0].cpu().tolist())
                                frame_labels.extend(labels.cpu().tolist())
                                # pre = torch.mean(outputs[:, 1])
                                # video_predictions.append(pre.cpu().item())
                                # video_labels.append(labels.cpu().item())
                            frame_results = get_test_metrics(y_pred=np.array(frame_predictions),
                                                             y_true=np.array(frame_labels),
                                                             img_names=valid_loader[key].dataset.image_list)
                            # frame_results = cal_metrics(frame_labels, frame_predictions, threshold=0.5)
                            # video_results = cal_metrics(video_labels, video_predictions, threshold=0.5)  # 'best' 'auto' or float
                            # log.info(
                            #     '{} result: Epoch [{:0>3}/{:0>3}], V_Acc: {:.2%}, V_Auc: {:.4} V_EER:{:.2%} F_Acc: {:.2%}, F_Auc: {:.4} F_EER:{:.2%}'
                            #     .format(key, epoch + 1, args.epochs, video_results.ACC, video_results.AUC, video_results.EER, frame_results.ACC,
                            #             frame_results.AUC, frame_results.EER))
                            if key != 'FaceForensics++':
                                avg_auc.append(frame_results.AUC)
                                avg_acc.append(frame_results.ACC)
                                avg_eer.append(frame_results.EER)
                                avg_vauc.append(frame_results.video_auc)
                            log.info(
                                'Epoch [{:0>3}/{:0>3}] {:18}: | Acc: {:.4} | Auc: {:.4} | EER:{:.4} | VAuc:{:.4} |'
                                .format(epoch + 1, args.epochs, key, frame_results.ACC, frame_results.AUC,
                                        frame_results.EER, frame_results.video_auc))
                    log.info('Epoch [{}] Step {}: | Avg_Acc: {:.4} | Avg_Auc: {:.4} Avg_EER:{:.4} Avg_VAuc:{:.4}'.format(epoch + 1, global_step,
                                                                                            np.mean(avg_acc),
                                                                                            np.mean(avg_auc),
                                                                                            np.mean(avg_eer),
                                                                                            np.mean(avg_vauc)
                                                                                            ))
                    if np.mean(avg_auc) > max_performance:
                        max_performance = np.mean(avg_auc)
                        torch.save(state, os.path.join(save_dir, 'models_params_best_{}.pth'.format(global_step)))
                        log.info('save best model to save_dir {} on step {}'.format(save_dir, global_step))
                    model.train()
                scheduler.step(global_step)


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        config = config.copy()
        config['test_dataset'] = test_name
        if config.get('test_dataset_type', None) is None or config['test_dataset_type'] == 'abs':
            test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test',
            )
        else:
            test_set = eval(config['test_dataset_type'])(
                config=config,
                mode='test',
            )

        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False,
            )

        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def calculate_parameters(model, logger):
    tot_params = 0
    train_params = 0
    for name, param in model.named_parameters():
        tot_params += param.numel()
        if param.requires_grad is True:
            logger.info(f'{name} is trainable')
            train_params += param.numel()
    print('Number of parameters: {} | Trainable parameters: {}'.format(tot_params, train_params))
    return tot_params, train_params


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--config_file', '-cf', type=str, default='configs/vit_moev1_0_0.yaml',
    parser.add_argument('--config_file', '-cf', type=str,
                        default='configs/clip_moev4_1_4_0_7_repeat_k1_n6_g2_s1_888161632_81632486496128_1e-4_3e-5_842.yaml',
                        help="which config file")
    parser.add_argument('--resume', '-r', type=int, default=960000, help="which epoch continue to train")
    parser.add_argument('--epochs', '-e', type=int, default=15)
    parser.add_argument('--record_step', '-rs', type=int, default=6400)
    parser.add_argument('--test_step', '-ts', type=int, default=32,
                        help="the iteration number to test state")

    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--num_frames', type=int, default=1)
    parser.add_argument('--num_workers', '-nw', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--loss_weight', type=str, default='[1, 1]')
    parser.add_argument('--dist', type=str, default='False')
    args = parser.parse_args()
    setup_seed(args.seed)
    print('Seed: {}'.format(args.seed))

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    print(config['log_dir'])
    config['log_dir'] = config['log_dir'] + '_' + str(args.seed) + '_' + str(args.batch_size)

    if args.dist == 'True':
        print('use distributed training')
        config['log_dir'] = config['log_dir'] + '_dist'
        init_dist('pytorch')
    args.rank, args.world_size = get_dist_info()
    device = torch.device('cuda:{}'.format(args.rank))
    mkdirs(config['log_dir'])
    with open(os.path.join(config['log_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f'save config file to {os.path.join(config["log_dir"], "config.yaml")}')
    args.model_dir = os.path.join(config['log_dir'], 'train')
    mkdirs(args.model_dir)

    train_dataset = eval(config['train_dataset_type'])(config, mode='train')
    if args.dist == 'True':
        print(args.batch_size//args.world_size)
        sampler = DistributedSampler(train_dataset, rank=args.rank, num_replicas=args.world_size, shuffle=True)
        train_loader = \
            torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=args.batch_size//args.world_size,
                num_workers=args.num_workers//args.world_size,
                collate_fn=train_dataset.collate_fn,
                sampler=sampler
            )
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  collate_fn=train_dataset.collate_fn)

    print(f'Rank: {args.rank} World_size: {args.world_size}')
    logger = None
    if args.rank == 0:
        # logging
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M')
        logger = logging.getLogger()
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        file_handler = logging.FileHandler(os.path.join(config['log_dir'], 'train.log'), mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info('log model dir:' + config['log_dir'])
        logger.info(f'batch size:{args.batch_size}')

    valid_loader = prepare_testing_data(config)
    clip_model, preprocess = clip.load('ViT-B/16', device=device)
    encode_text_func = clip_model.encode_text
    if config['model_name'] == 'vit_moe':
        print('load vit moe')
        model = vit_base_patch16_224_in21k(pretrained=True, num_classes=2)
    else:
        import importlib
        M = getattr(importlib.import_module('networks.' + config['model_name'], package=None), config['model_name'])
        if config['model_name'].startswith('ClipMoev2'):
            model = M(config=config, encode_text_func=encode_text_func)
        else:
            model = M(config=config, clip_model=clip_model)
    if logger is not None:
        logger.info('Model Name: {}'.format(config['model_name']))
    model = model.to(device)
    if args.dist == 'True':
        model = DistributedDataParallel(model, device_ids=[args.rank],
                                      find_unused_parameters=True)
    if args.rank == 0:
        calculate_parameters(model, logger)
    # special defined optim
    special_param = []
    other_param = []
    for name, param in model.named_parameters():
        if 'w_gate' in name or 'w_noise' in name:
            special_param.append(param)
        else:
            other_param.append(param)
    if config['optimizer']['type'] == 'adam':
        optimizer = optim.AdamW(
            [
                {'params': special_param, 'lr': float(config['optimizer']['adam']['lr'])*args.batch_size/32,
                 'initial_lr': float(config['optimizer']['adam']['lr'])*args.batch_size/32},
                {'params': other_param, 'lr': float(config['optimizer']['adam']['other_lr'])*args.batch_size/32,
                 'initial_lr': float(config['optimizer']['adam']['other_lr'])*args.batch_size/32}
            ],
            betas=(0.9, 0.999), weight_decay=1e-5)
    elif config['optimizer']['type'] == 'adamw':
        optimizer = optim.AdamW(
            [
                {'params': special_param, 'lr': float(config['optimizer']['adam']['lr'])*args.batch_size/32,
                 'initial_lr': float(config['optimizer']['adam']['lr'])*args.batch_size/32},
                {'params': other_param, 'lr': float(config['optimizer']['adam']['other_lr'])*args.batch_size/32,
                 'initial_lr': float(config['optimizer']['adam']['other_lr'])*args.batch_size/32}
            ],
            betas=(0.9, 0.999), weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_scheduler']['step_size'], gamma=0.5,
                                                last_epoch=args.resume)

    print('Start train process...')
    train(args, model, optimizer, train_loader, valid_loader, scheduler, logger)
    print('Finish train process...')
