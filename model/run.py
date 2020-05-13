from __future__ import division
from __future__ import print_function

import torch
from gcn.models import *
from rl.models import *
from rl.replays_buffer import *
from rl.utils import *
from absl import flags
import numpy as np
import random
import sys


def get_candidate_ids(labels, n):
    candidate_list = []

    for i in range(len(labels)):
        if labels[i] == 1:
            candidate_list.append(i)

    remain_list = list(set(range(len(labels))).difference(set(candidate_list)))
    list2 = random.sample(remain_list, len(candidate_list)*n)

    candidate_list.extend(list2)
    return candidate_list


def model_train(dataset):
    # Load data
    adj_norm_1, adj_norm_2, adj_1, adj_2, features, labels, candidate_list = load_data_gcn(dataset)
    num_nodes = adj_norm_1[2][0]
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create Model
    model_rgcn_main = RGCN(num_features, features_nonzero).to(device)
    model_rgcn_target = RGCN(num_features, features_nonzero).to(device)

    state_representations_main = model_rgcn_main(adj_1, adj_2, 0.5, torch.sparse_coo_tensor(torch.tensor(features[0].transpose()), torch.tensor(features[1]), features[2])).to(device)
    state_representations_target = model_rgcn_target(adj_1, adj_2, 0.5, torch.sparse_coo_tensor(torch.tensor(features[0].transpose()), torch.tensor(features[1]), features[2])).to(device)

    model_rl_target = DQN(state=state_representations_target,
                          output_dim=num_nodes).to(device)
    model_rl_main = DQN(state=state_representations_main,
                        output_dim=num_nodes).to(device)

    parameters = list(model_rgcn_main.parameters()) + list(model_rl_main.parameters())

    replay_buffer = ReplayBuffer(buffer_size=FLAGS.buffer_size)
    frame_count = 0
    gcn_params = {'adj_norm_1': adj_norm_1,
                  'adj_norm_2': adj_norm_2,
                  'adj_1': adj_1,
                  'adj_2': adj_2,
                  'features': features,
                  'labels': labels}

    for epoch in range(FLAGS.epochs):
        episode_reward, episode_loss, frame_count = run_training_episode(model_rgcn_main = model_rgcn_main,
                                                        model_rgcn_target = model_rgcn_target,
                                                        model_rl_main=model_rl_main,
                                                        model_rl_target=model_rl_target,
                                                        gcn_params=gcn_params,
                                                        replay_buffer=replay_buffer,
                                                        frame_count=frame_count,
                                                        candidate_ids=candidate_list,
                                                        parameters = parameters)
        print("Epoch: {}, Reward: {}, Loss: {}".format(epoch, episode_reward, episode_loss))
        if epoch % 50 == 0:
              torch.save(model_rl_main, FLAGS.model_path + '/model' + str(epoch) + '.cptk')


if __name__ == '__main__':
    # Set random seed
    seed = 123
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Settings
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
    flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')

    flags.DEFINE_integer('buffer_size', 10000, 'The maximum size of the replay buffer for DQN.')
    flags.DEFINE_integer('rl_episode_max_steps', 50, 'The maximum steps for rl agent')
    flags.DEFINE_multi_float('epsilon', [1, 1000, 0.1],
                             ['Initial exploration rate', 'anneal steps', 'final exploration rate'])
    flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
    flags.DEFINE_integer('replay_start_size', 20, 'Number of experiences to be stored before training.')
    flags.DEFINE_integer('target_update_freq', 100, 'rl target network update frequency.')
    flags.DEFINE_integer('main_update_freq', 10, 'rl main network update frequency.')
    flags.DEFINE_integer('rl_batch_size', 10, 'Batch size for training rl.')
    flags.DEFINE_float('rl_lr', 0.0001, 'Initial rl learning rate.')

    flags.DEFINE_string('summary_path', './log', 'Path to store training summary.')
    flags.DEFINE_string('model_path', './model', 'Path to store model.')

    dataset = "BlogCatalog_classification"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device='cpu'
    print('device:', device)

    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    model_train(dataset)