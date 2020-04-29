from __future__ import division
from __future__ import print_function

from gcn.utils import *
import numpy as np
import sys
import torch.optim as optim
sys.path.append('..')

flags = tf.app.flags
FLAGS = flags.FLAGS


def get_reward_simple(selected_list, labels):
    count = 0
    for id in selected_list:
        if labels[id] == 1:
            count += 1

    return count / np.sum(labels)

 def to_one_hot_vector(,label):
            b = np.zeros((label.shape[0], num_class))
            b[np.arange(label.shape[0]), label] = 1
            return b


def update_main_rl(sess,
                   model_rl_target,
                   model_rl_main,
                   replay_buffer):
    #[0:s, 1:a, 2:r, 3:s', 4:done]
   # with sess. as_default():
       train_batch = replay_buffer.sample(size=FLAGS.rl_batch_size)
        # train_batch = np.moveaxis(train_batch, 0, 1)
       # next_q_prime = sess.run(model_rl_target.qvalues,
       #                         feed_dict=train_batch[0, 3])
       
       next_q_prime = model_rl_target(model_rl_target.state)
       # next_q = sess.run(model_rl_main.qvalues,
       #                   feed_dict=train_batch[0, 3])
       
       next_q = model_rl_main(model_rl_main.state)
       
        target_qvalues = train_batch[0, 2] + (1-train_batch[0, 4])*FLAGS.gamma*next_q_prime[0, np.argmax(next_q[0])]
        print(target_qvalues)
        #feed_dict = train_batch[0, 0]
        #feed_dict.update({model_rl_main.chosen_actions: [train_batch[0, 1]],
        #                  model_rl_main.target_qvalues: [target_qvalues],
        #                  model_rl_main.lr: FLAGS.rl_lr})
        
        chosen_actions = [train_batch[0, 1]]
        target_qvalues = [target_qvalues]
        lr = FLAGS.rl_lr
        
        
       actions_onehot = to_one_hot_vector(chosen_actions, train_batch[0][0]['adj_norm_1'][2][0])
       # actions_onehot = to_one_hot_vector(self.chosen_actions, output_dim)
       qvalues_for_chosen_actions = torch.sum(next_q*actions_onehot, axis=1)
       td_error = torch.mul(target_qvalues-qvalues_for_chosen_actions)
       loss = 0.5 * td_error
       
       loss.backward()
       
       optimizer = optim.RMSprop([model_rl_main.parameters(), model_rgcn_main.parameters()], lr=lr)
       optimizer.step()
       
       optimizer.zero_grad()
       
       loss = loss.item()
       
       

       # params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        #gradients = tf.gradients(self.loss, params)
        #norm_gradients, _ = tf.clip_by_global_norm(gradients, 40.0)
        #trainer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        #self.update = trainer.apply_gradients(zip(norm_gradients, params))
        
        # feed_dict[model_rl_main.chosen_actions] = np.vstack(train_batch[0, 1])
        # feed_dict[model_rl_main.target_qvalues] = target_qvalues
        # feed_dict[model_rl_main.lr] = FLAGS.rl_lr
        # loss, _ = sess.run([model_rl_main.loss,
        #                    model_rl_main.update],
        #                   feed_dict=feed_dict)
        return loss


def get_target_update_op():
    target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target')
    main_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'main')
    op_holder = []
    for tp, mp in zip(target_params, main_params):
        op_holder.append(tp.assign(mp.value()))
    return op_holder


def run_training_episode(sess,
                         model_rl_target,
                         model_rl_main,
                         gcn_params,
                         replay_buffer,
                         #target_update_op,
                         frame_count,
                         candidate_ids
                         ):
    episode_reward = 0
    episode_losses = []
    selected_list = []
   # feed_dict = construct_feed_dict(gcn_params['adj_norm_1'],
   #                                 gcn_params['adj_norm_2'],
   #                                 gcn_params['features'],
   #                                 FLAGS.dropout,
   #                                 gcn_params['placeholders'])
    # state = sess.run(model_gcn.outputs,
    #                  feed_dict=feed_dict)
    steps = 0
    count = 0
    while steps < FLAGS.rl_episode_max_steps:
        qvalues = model_rl_main() 
        candidate_ids = list(set(candidate_ids).difference(set(selected_list)))
        qvalues_masked = qvalues[candidate_ids]
        # print(qvalues_masked[:10])
        # print(qvalues[3041])

        (ep_start, anneal_steps, ep_end) = FLAGS.epsilon
        ratio = max((anneal_steps - max(frame_count-FLAGS.replay_start_size, 0))/float(anneal_steps), 0)
        ep = (ep_start - ep_end)*ratio + ep_end

        print("Epsison: {}".format(ep))
        selected_node_id = candidate_ids[np.random.choice(len(qvalues_masked))] \
            if np.random.rand() < ep else candidate_ids[np.argmax(qvalues_masked)]
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
                  'placeholders': gcn_params['placeholders'],
                  'labels': gcn_params['labels']}
         
        new_gcn_params['adj_norm_1'], new_gcn_params['adj_norm_2'], new_gcn_params['adj_1'], new_gcn_params['adj_2'] \
            = update_adj(selected_node_id, gcn_params['adj_1'], gcn_params['adj_2'])


        episode_reward += r

        # check if done and get new state
        done = True if steps == FLAGS.rl_episode_max_steps-1 else False

      #  new_feed_dict = construct_feed_dict(gcn_params['adj_norm_1'],
      #                                      gcn_params['adj_norm_2'],
      #                                      gcn_params['features'],
      #                                     FLAGS.dropout,
      #                                       gcn_params['placeholders'])

        replay_buffer.add(np.reshape(np.array([ gcn_params, selected_node_id, r, new_gcn_params, done]), [1, -1]))

        if frame_count > FLAGS.replay_start_size:
            if frame_count % FLAGS.main_update_freq == 0:
                loss = update_main_rl(sess=sess,
                                      model_rl_target=model_rl_target,
                                      model_rl_main=model_rl_main,
                                      replay_buffer=replay_buffer)
                episode_losses.append(loss)
            if frame_count % FLAGS.target_update_freq == 0:
                #sess.run(target_update_op)
                model_rl_target.load_state_dict(model_rl_main.state_dict())

        gcn_params = new_gcn_params
        frame_count += 1
        steps += 1

        # print(count)
        if done:
            break

    episode_loss = np.mean(episode_losses) if len(episode_losses) != 0 else 0
    return episode_reward, episode_loss, frame_count