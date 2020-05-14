from __future__ import division
from __future__ import print_function

from gcn.utils import *
import numpy as np
import sys
import torch.optim as optim
import torch
import torch.nn as nn
from absl import flags
sys.path.append('..')

FLAGS = flags.FLAGS

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('rl_util', device)

def get_reward_simple(selected_list, labels):
    count = 0
    for id in selected_list:
        if labels[id] == 1:
            count += 1

    return count / np.sum(labels)

def to_one_hot_vector(label, num_class):
            b = np.zeros((len(label), num_class))
            b[np.arange(len(label)), label] = 1
            return b


def update_main_rl(model_rgcn_main,
                   model_rgcn_target,
                   model_rl_main,
                   model_rl_target,
                   replay_buffer,
                   parameters):
    #[0:s, 1:a, 2:r, 3:s', 4:done]
       train_batch = replay_buffer.sample(size=FLAGS.rl_batch_size)

       rewards = []
       state_rep_main = []
       state_rep_target = []
       actions = []

       for idx in enumerate(train_batch):
           state_rep_main.append(model_rgcn_main(train_batch[idx[0],3]['adj_1'],
                                            train_batch[idx[0], 3]['adj_2'],
                                            0.5,
                                            torch.sparse_coo_tensor(torch.tensor(train_batch[idx[0], 3]['features'][0].transpose()),
                                                                    torch.tensor(train_batch[idx[0], 3]['features'][1]),
                                                                    train_batch[idx[0], 3]['features'][2])))
           state_rep_target.append(model_rgcn_target(train_batch[idx[0], 3]['adj_1'],
                                                train_batch[idx[0], 3]['adj_2'],
                                                0.5,
                                                torch.sparse_coo_tensor(torch.tensor(train_batch[idx[0], 3]['features'][0].transpose()),
                                                                        torch.tensor(train_batch[idx[0], 3]['features'][1]),
                                                                        train_batch[idx[0], 3]['features'][2])))

           rewards.append(train_batch[idx[0], 2])
           actions.append(train_batch[idx[0], 1])

       state_rep_main = torch.cat(state_rep_main, 0).to(device)
       reward = torch.FloatTensor(rewards).to(device)
       action = torch.LongTensor(actions).to(device)
       state_rep_target = torch.cat(state_rep_target, 0).to(device)

       action = action.unsqueeze(1).to(device)
       next_q = model_rl_main(state_rep_main)
       next_q = next_q.gather(1, action)
       print('next_q:',next_q[0][:5])
       next_q_prime = model_rl_target(state_rep_target)
       next_q_prime = next_q_prime.detach()


       target_qvalues = (reward+FLAGS.gamma*next_q_prime.max(1)[0]).unsqueeze(1)

       print('updating RL')

       loss = nn.MSELoss()
       loss = loss(next_q, target_qvalues)

       lr = FLAGS.rl_lr
       optimizer = optim.RMSprop(parameters, lr=lr)
       optimizer.zero_grad()
       loss.backward(retain_graph=True)
       optimizer.step()

       return loss

def run_training_episode(model_rgcn_main,
                         model_rgcn_target,
                         model_rl_main,
                         model_rl_target,
                         gcn_params,
                         replay_buffer,
                         frame_count,
                         candidate_ids,
                         parameters
                         ):
    episode_reward = 0
    episode_losses = []
    selected_list = []
    steps = 0
    count = 0
    while steps < FLAGS.rl_episode_max_steps:
        features = gcn_params['features']
        adj_1 = gcn_params['adj_1']
        adj_2 = gcn_params['adj_2']
        state_rep_main = model_rgcn_main(adj_1, adj_2, 0.5, torch.sparse_coo_tensor(torch.tensor(features[0].transpose()), torch.tensor(features[1]), features[2])).to(device)
     #   state_rep_target = model_rgcn_target(adj_1, adj_2, 0.5, torch.sparse_coo_tensor(torch.tensor(features[0].transpose()), torch.tensor(features[1]), features[2])).to(device)
        qvalues = model_rl_main(state_rep_main).to(device)
        candidate_ids = list(set(candidate_ids).difference(set(selected_list)))
        qvalues_masked = qvalues[0][candidate_ids]

        (ep_start, anneal_steps, ep_end) = FLAGS.epsilon
        ratio = max((anneal_steps - max(frame_count-FLAGS.replay_start_size, 0))/float(anneal_steps), 0)
        ep = (ep_start - ep_end)*ratio + ep_end

        print("Epsilon: {}".format(ep))
        selected_node_id = candidate_ids[np.random.choice(len(qvalues_masked))] \
            if np.random.rand() < ep else candidate_ids[torch.argmax(qvalues_masked)]
        print(selected_node_id)
        r = get_reward_simple(selected_list, gcn_params['labels']) if steps == FLAGS.rl_episode_max_steps - 1 else 0
        if selected_node_id in selected_list:
            r -= 0.01
            count += 1

        selected_list.append(selected_node_id)
        
        new_gcn_params = {'adj_norm_1': gcn_params['adj_norm_1'],
                  'adj_norm_2': gcn_params['adj_norm_2'],
                  'adj_1': gcn_params['adj_1'],
                  'adj_2': gcn_params['adj_2'],
                  'features': gcn_params['features'],
                  'labels': gcn_params['labels']}
         
        new_gcn_params['adj_norm_1'], new_gcn_params['adj_norm_2'], new_gcn_params['adj_1'], new_gcn_params['adj_2'] \
            = update_adj(selected_node_id, gcn_params['adj_1'], gcn_params['adj_2'])


        episode_reward += r

        # check if done and get new state
        done = True if steps == FLAGS.rl_episode_max_steps-1 else False

        replay_buffer.add(np.reshape(np.array([ gcn_params, selected_node_id, r, new_gcn_params, done]), [1, -1]))

        if frame_count > FLAGS.replay_start_size:
            if frame_count % FLAGS.main_update_freq == 0:
                loss = update_main_rl(model_rgcn_main = model_rgcn_main,
                                      model_rgcn_target = model_rgcn_target,
                                      model_rl_main=model_rl_main,
                                      model_rl_target=model_rl_target,
                                      replay_buffer=replay_buffer,
                                      parameters = parameters)
                episode_losses.append(loss)
            if frame_count % FLAGS.target_update_freq == 0:
                model_rl_target.load_state_dict(model_rl_main.state_dict())

        gcn_params = new_gcn_params
        frame_count += 1
        steps += 1

        print(count)
        if done:
            break

    episode_loss = torch.mean(torch.FloatTensor(episode_losses)) if len(episode_losses) != 0 else 0
    return episode_reward, episode_loss, frame_count