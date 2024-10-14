# coding: utf-8

from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


def quick_start(model, dataset, config_dict, save_model=True):
    # merge config dict
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx + 1, total_loops, config['hyper_parameters'], hyper_tuple))

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        logger.info(model)

        #
        # # Step 4: Convert the sampled features to a PyTorch tensor
        # sampled_features_tensor = torch.tensor(sampled_features, dtype=torch.float32)
        #
        # # Step 5: Extract feature representations using the model
        # with torch.no_grad():  # Disable gradient computation since we're only doing inference
        #     feature_representations = model(sampled_features_tensor).numpy()

        # features = np.load('image_feat.npy')

        model.load_state_dict(torch.load('FREEDOM_mamba_baby.pth'))

        # features_tensor = torch.tensor(features, dtype=torch.float32)

        # 使用模型提取特征表示
        with torch.no_grad():
            output = model()  # Call the model

        np.save('mamba_visual', output.detach().cpu().numpy())

        # Step 6: Apply t-SNE to reduce dimensionality to 2D
        # tsne = TSNE(n_components=2, random_state=42)
        #
        # # Apply t-SNE to raw features
        # tsne_raw = tsne.fit_transform(sampled_features)
        # # tsne_raw_normalized = normalize(tsne_raw)
        #
        # # Apply t-SNE to purified features
        # tsne_refined = tsne.fit_transform(feature_representations)
        # # tsne_refined_normalized = normalize(tsne_refined)
        #
        # # Step 3: Visualization (scatter plots and angle density plots)
        # plt.figure(figsize=(12, 8))
        #
        # # Scatter plot for raw features
        # plt.subplot(2, 2, 1)
        # plt.scatter(tsne_raw[:, 0], tsne_raw[:, 1], c=np.random.rand(len(sampled_features_tensor)), alpha=0.7)
        # plt.title('Raw Feature')
        # plt.xlabel('Features')
        # plt.ylabel('Distribution')
        # plt.axis('equal')
        #
        # # Scatter plot for purified features
        # plt.subplot(2, 2, 2)
        # plt.scatter(tsne_refined[:, 0], tsne_refined[:, 1], c=np.random.rand(len(feature_representations)), alpha=0.7)
        # plt.title('Refined Feature')
        # plt.xlabel('Features')
        # plt.ylabel('Distribution')
        # plt.axis('equal')
        #
        # # Step 4: Calculate angles for both t-SNE results
        # # angles_raw = np.arctan2(tsne_raw_normalized[:, 1], tsne_raw_normalized[:, 0])
        # # angles_purified = np.arctan2(tsne_purified_normalized[:, 1], tsne_purified_normalized[:, 0])
        # #
        # # # Angle density plot for raw features
        # # plt.subplot(2, 2, 3)
        # # plt.hist(angles_raw, bins=50, density=True, alpha=0.7, color='orange')
        # # plt.title('Raw Feature Angles')
        # # plt.xlabel('Angles')
        # # plt.ylabel('Density')
        #
        # # # Angle density plot for purified features
        # # plt.subplot(2, 2, 4)
        # # plt.hist(angles_purified, bins=50, density=True, alpha=0.7, color='orange')
        # # plt.title('Purified Feature Angles')
        # # plt.xlabel('Angles')
        # # plt.ylabel('Density')
        #
        # # Adjust layout and display the final plot
        # plt.tight_layout()
        #
        # plt.savefig('lgmrec_mamba_visual_representation.png', format='png', dpi=300)
        # plt.show()

        # def extract_features(model, data):
        #     with torch.no_grad():
        #         output = model(data)  # 通过模型前向传播获取特征
        #     return output.cpu().numpy()
        #
        # input_data = torch.randn(10, 128)  # 假设输入特征维度为 128
        # baseline_features = extract_features(model, input_data)

        # trainer loading and initialization
        # trainer = get_trainer()(config, model)
        # debug
        # model training
        # best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        # #########
        # hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        # save best test
        # if best_test_upon_valid[val_metric] > best_test_value:
        #     best_test_value = best_test_upon_valid[val_metric]
        #     best_test_idx = idx
        # idx += 1

    #     logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
    #     logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
    #     logger.info('████Current BEST████:\nParameters: {}={},\n'
    #                 'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
    #         hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))
    #
    # # log info
    # logger.info('\n============All Over=====================')
    # for (p, k, v) in hyper_ret:
    #     logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
    #                                                                               p, dict2str(k), dict2str(v)))
    #
    # logger.info('\n\n█████████████ BEST ████████████████')
    # logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
    #                                                                hyper_ret[best_test_idx][0],
    #                                                                dict2str(hyper_ret[best_test_idx][1]),
    #                                                                dict2str(hyper_ret[best_test_idx][2])))

    # import numpy as np
    # _, i_v_feats = model.agg_mm_neighbors('v')
    # np.save('log/'+config['model'] + '-' + config['dataset']+'-i_v_feats', i_v_feats.detach().cpu().numpy())
    # _, i_t_feats = model.agg_mm_neighbors('t')
    # np.save('log/'+config['model'] + '-' + config['dataset']+'-i_t_feats', i_t_feats.detach().cpu().numpy())

