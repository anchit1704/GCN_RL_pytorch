from __future__ import division
from __future__ import print_function

from gcn.utils import *
import numpy as np
import sys
import torch.optim as optim
import torch
from absl import flags
sys.path.append('..')

FLAGS = flags.FLAGS


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


def update_main_rl(state_rep_main,
                   state_rep_target,
                   model_rl_target,
                   model_rl_main,
                   replay_buffer,
                   parameters):
    #[0:s, 1:a, 2:r, 3:s', 4:done]
       train_batch = replay_buffer.sample(size=FLAGS.rl_batch_size)



       next_q_prime = []
       next_q = []
       for idx in enumerate(train_batch):
           next_q_prime.append(model_rl_target(state_rep_target))
           next_q.append(model_rl_main(state_rep_main))


       next_q = torch.cat(next_q, 0)
       next_q_prime = torch.cat(next_q_prime, 0)
       next_q_prime = next_q_prime.detach()




       print('updating RL')

       loss = []

       for idx in enumerate(train_batch):
           target_qvalues = train_batch[idx[0], 2] + (1-train_batch[idx[0], 4])*FLAGS.gamma*next_q_prime[idx[0], torch.argmax(next_q[idx[0]])]
           print(target_qvalues)

           chosen_actions = [train_batch[idx[0], 1]]
           target_qvalues = torch.DoubleTensor([target_qvalues])
           lr = FLAGS.rl_lr
        
        
           actions_onehot = torch.from_numpy(to_one_hot_vector(chosen_actions, train_batch[idx[0]][0]['adj_norm_1'][2][0])).type(torch.DoubleTensor)
           qvalues_for_chosen_actions = torch.sum(next_q[idx[0]].type(torch.DoubleTensor)*actions_onehot, axis=1)
           td_error = torch.mul(target_qvalues-qvalues_for_chosen_actions, target_qvalues-qvalues_for_chosen_actions)
           loss.append(0.5 * td_error)


       loss = torch.FloatTensor(loss)
       loss = torch.sum(loss)
       loss.requires_grad = True
       loss.backward(retain_graph = True)
       
       optimizer = optim.RMSprop(parameters, lr=lr)
       optimizer.step()
       
       optimizer.zero_grad()
       
       loss = loss.item()
       return loss

def run_training_episode(model_rgcn_main,
                         model_rgcn_target,
                         model_rl_target,
                         model_rl_main,
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
        state_rep_main = model_rgcn_main(adj_1, adj_2, 0.5, torch.sparse_coo_tensor(torch.tensor(features[0].transpose()), torch.tensor(features[1]), features[2]))
        state_rep_target = model_rgcn_target(adj_1, adj_2, 0.5, torch.sparse_coo_tensor(torch.tensor(features[0].transpose()), torch.tensor(features[1]), features[2]))
        qvalues = model_rl_main(state_rep_main)
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
                loss = update_main_rl(state_rep_main = state_rep_main,
                                      state_rep_target = state_rep_target,
                                      model_rl_target=model_rl_target,
                                      model_rl_main=model_rl_main,
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

    episode_loss = np.mean(episode_losses) if len(episode_losses) != 0 else 0
    return episode_reward, episode_loss, frame_count