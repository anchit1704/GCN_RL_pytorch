import torch
import torch.nn as nn
import tensorflow.contrib.slim as slim

class DQN(nn.Module):
    def __init__(self, state, output_dim):
        super(DQN, self).__init__()
       # with tf.variable_scope(scope):
            # self.inputs = tf.placeholder(shape=[None, input_dim], dtype=tf.float32)
       # self.hidden1 = slim.fully_connected(inputs=state,
       #                                     num_outputs=512,
        #                                    activation_fn=tf.nn.tanh,
        #                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
        #                                    biases_initializer=tf.zeros_initializer(),
        #                                    scope='hidden1')
        
        self.state = state

        self.hidden1 = torch.nn.Linear(state.shape[1], 512, bias=True)
        self.tanh = torch.nn.Tanh()


        
        self.hidden2 = torch.nn.Linear(512, 256, bias=True)

        
        self.qvalues = torch.nn.Linear(256, output_dim, bias=True)

       
       
    def forward(self,):
           h1 = self.hidden1(self.state)
           h1 = self.tanh(h1)
           h2 = self.hidden2(h1)
           h2 = self.tanh(h2)
           qvalues = self.qvalues(h2)
           qvalues = self.tanh(qvalues)
           return qvalues
           
       
      
       
        # training
       # self.chosen_actions = tf.placeholder(shape=[None], dtype=tf.int32, name="chosen_actions")
       # self.target_qvalues = tf.placeholder(shape=[None], dtype=tf.float32, name="target_qvalues")
       # self.lr = tf.placeholder(dtype=tf.float32, name="lr")

        #actions_onehot = tf.one_hot(self.chosen_actions, output_dim, dtype=tf.float32)
       # actions_onehot = to_one_hot_vector(self.chosen_actions, output_dim)
       # qvalues_for_chosen_actions = torch.sum(self.qvalues*actions_onehot, axis=1)
       # td_error = tf.square(self.target_qvalues-qvalues_for_chosen_actions)
       # self.loss = 0.5 * td_error

       # params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        #gradients = tf.gradients(self.loss, params)
        #norm_gradients, _ = tf.clip_by_global_norm(gradients, 40.0)
        #trainer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        #self.update = trainer.apply_gradients(zip(norm_gradients, params))