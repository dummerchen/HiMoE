## -*- coding: utf-8 -*-
import sys

import numpy as np
import yaml
from datasets import *
import clip
sys.setrecursionlimit(15000)
import random
from torch.utils.data import DataLoader
import time
from utils import *
from networks import *
import logging


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def test(args, model, test_loader):
    device = args.device
    checkpoint = torch.load(os.path.join(args.model_dir, 'train', 'models_params_{}.pth'.format(args.resume)), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    log.info('Loading from models_params_{} Start Test'.format(args.resume))
    model.eval()
    with torch.no_grad():
        keys = test_loader.keys()
        print('Valid Loader Datasets:', keys)
        for key in keys:
            frame_predictions = []
            frame_labels = []
            data_dict =test_loader[key].dataset.data_dict
            for data in tqdm(test_loader[key], total=len(test_loader[key])):
                inputs, labels = data['aug_image'], data['labels']
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs, _ = model(inputs)
                outputs = F.softmax(outputs, dim=-1)
                frame = outputs.shape[0]
                frame_predictions.extend(np.ones_like(outputs[:, 0].cpu())-outputs[:, 0].cpu().tolist())
                frame_labels.extend(labels.expand(frame).cpu().tolist())

            if 'abs' in config['test_dataset_type'].lower():
                frame_results = get_test_metrics(y_true=np.array(frame_labels), y_pred=np.array(frame_predictions), img_names=test_loader[key].dataset.image_list)
                # video_results = cal_metrics(video_labels, video_predictions, threshold=0.5)  # 'best' 'auto' or float
                log.info('{} Test result: Acc: {:.3%} F_VAuc: {:.3%}, F_Auc: {:.3%} F_EER:{:.3%}'
                         .format(key, frame_results.ACC, frame_results.video_auc, frame_results.AUC, frame_results.EER))
            else:
                prediction_class = (np.array(frame_predictions) > 0.5).astype(int)
                correct = (prediction_class == np.clip(np.array(frame_labels), a_min=0, a_max=1)).sum().item()
                acc = correct / len(prediction_class)
                log.info('{} Test result: Acc: {:.2%}'.format(key, acc))

def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        if config.get('test_dataset_type', None) is None or 'abs' in config['test_dataset_type'].lower():
            print('Loading Deepfake Abstract Base Dataset {}'.format(config['test_dataset_type']))
            test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test',
            )
        else:
            print('Loading Diffusion Dataset {}'.format(config['test_dataset_type']))
            test_set = DiffusionDataset(
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', type=str, default="cuda:0", help="specify which GPU to use")
    parser.add_argument('--config_file', '-cf', type=str, default='configs/clip_moev4_1_4_0_7_repeat_k1_n6_g2_s1_888161632_81632486496128_1e-4_3e-5_842.yaml')
    parser.add_argument('--resume', '-r', type=int, default=960000, help="which epoch to use for evaluation")
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--test_dataset', '-td', nargs='+', default=[None])
    # parser.add_argument('--test_dataset', '-da', type=list, default=['DeepFakeDetection'])
    # parser.add_argument('--test_dataset', '-da', type=list, default=[
    #     'blendface_cdf', 'starganv2'])
    parser.add_argument('--test_dataset_type', '-tat', type=str, default='None', help='None is abs or diffusion')
    parser.add_argument('--num_frames', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--test_level', '-tl', type=int, default=0,help='0: no data augmentation, 1-4 use data augmentation test')
    parser.add_argument('--test_data_aug', '-tda', nargs='+', default=[], help='CS,CC,BW, GNC(gaussian noise color), GB, JEPG')
    # python3 eval -cf xxx -tda CC -tl 1 -td DFDC -r 832000
    
    args = parser.parse_args()

    start_time = time.time()
    setup_seed(args.seed)
    device = args.device
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    with open('./configs/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if args.test_level != 0:
        config['use_test_data_augmentation'] = True
        config['test_level'] = args.test_level
        config['test_data_aug'] = args.test_data_aug

    args.model_dir = f'{config["log_dir"]}_{str(args.seed)}_{config["train_batchSize"]}'
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(args.model_dir, 'test.log'), mode='a')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.info('model dir:' + args.model_dir)
    if args.test_dataset[0] is not None:
        config['test_dataset'] = args.test_dataset
    if args.test_dataset_type != 'None':
        config['test_dataset_type'] = args.test_dataset_type
    test_loader = prepare_testing_data(config)

    clip_model, preprocess = clip.clip.load('ViT-B/16')
    clip_model = clip_model.to(device)
    encode_text_func = clip_model.encode_text
    if config['model_name'] == 'vit_moe':
        print('load vit moe')
        model = vit_base_patch16_224_in21k(pretrained=True, num_classes=2)
    else:
        import importlib
        M = getattr(importlib.import_module('networks.' + config['model_name'], package=None), config['model_name'])
        model = M(config=config, clip_model=clip_model)
    model = model.to(device)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    print('Start eval process...')
    test(args, model, test_loader)
