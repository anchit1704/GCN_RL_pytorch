import torch
import tensorflow.contrib.slim as slim

class DQN():
    def __init__(self, state, output_dim, scope):
       # with tf.variable_scope(scope):
            # self.inputs = tf.placeholder(shape=[None, input_dim], dtype=tf.float32)
       # self.hidden1 = slim.fully_connected(inputs=state,
       #                                     num_outputs=512,
        #                                    activation_fn=tf.nn.tanh,
        #                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
        #                                    biases_initializer=tf.zeros_initializer(),
        #                                    scope='hidden1')
        
        self.hidden1 = torch.nn.Sequential(
        torch.nn.Linear(state, 512, bias=True),
        torch.nn.Tanh(),
        )
        
        self.hidden2 = torch.nn.Sequential(
        torch.nn.Linear(self.hidden1, 256, bias=True),
        torch.nn.Tanh(),
        ) 
        
        self.qvalues =  torch.nn.Sequential(
        torch.nn.Linear(self.hidden2, output_dim, bias=True),
        torch.nn.Tanh(),
        )

      #  self.hidden2 = slim.fully_connected(inputs=self.hidden1,
      #                                      num_outputs=256,
      #                                      activation_fn=tf.nn.tanh,
      #                                      weights_initializer=tf.contrib.layers.xavier_initializer(),
      #                                      biases_initializer=tf.zeros_initializer(),
      #                                      scope='hidden2')
      
       # self.qvalues = slim.fully_connected(inputs=self.hidden2,
       #                                     num_outputs=output_dim,
       #                                     activation_fn=tf.nn.tanh,
       #                                     weights_initializer=tf.contrib.layers.xavier_initializer(),
       #                                     biases_initializer=tf.zeros_initializer(),
       #                                     scope='qvalues')
       
       def to_one_hot_vector(,label):
            b = np.zeros((label.shape[0], num_class))
            b[np.arange(label.shape[0]), label] = 1
            return b
       
        # training
        self.chosen_actions = tf.placeholder(shape=[None], dtype=tf.int32, name="chosen_actions")
        self.target_qvalues = tf.placeholder(shape=[None], dtype=tf.float32, name="target_qvalues")
        self.lr = tf.placeholder(dtype=tf.float32, name="lr")

        #actions_onehot = tf.one_hot(self.chosen_actions, output_dim, dtype=tf.float32)
        actions_onehot = to_one_hot_vector(self.chosen_actions, output_dim)
        qvalues_for_chosen_actions = torch.sum(self.qvalues*actions_onehot, axis=1)
        td_error = tf.square(self.target_qvalues-qvalues_for_chosen_actions)
        self.loss = 0.5 * td_error

       # params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        #gradients = tf.gradients(self.loss, params)
        #norm_gradients, _ = tf.clip_by_global_norm(gradients, 40.0)
        #trainer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        #self.update = trainer.apply_gradients(zip(norm_gradients, params))