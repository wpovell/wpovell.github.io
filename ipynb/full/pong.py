import os
import gym
import numpy as np
import tensorflow as tf
from full.misc import DataSaver

IMAGE_SIZE = 80 * 80
HIDDEN_SIZE = 200
N_ACTIONS = 3
LEARNING_RATE = 1e-3
GAMMA = 0.99
NAME = 'pong'

def preprocess(I):
    """ Process 210x160x3 uint8 frame into 6400 (80x80) 1D float vector.
    """
    I = I[35:195]    # Crop
    I = I[::2,::2,0] # Downsample by factor of 2
    I[I == 144] = 0  # Erase background (background type 1)
    I[I == 109] = 0  # Erase background (background type 2)
    I[I != 0] = 1    # Everything else (paddles, ball) set to 1
    return I.astype(np.float).ravel() # Flatten

def train():
    dataSaver = DataSaver(NAME)

    # Make our game
    env = gym.make("Pong-v0")
    observation = env.reset()

    # Input image
    X = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE],name="X")
    # Output action
    Y = tf.placeholder(dtype=tf.float32, shape=[None, N_ACTIONS],name="Y")

    # Weight vectors
    W1 = tf.get_variable("W1", [IMAGE_SIZE, HIDDEN_SIZE],
                         initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [HIDDEN_SIZE, N_ACTIONS],
                         initializer=tf.contrib.layers.xavier_initializer())

    hidden = tf.nn.relu(tf.matmul(X, W1))
    logits = tf.matmul(hidden, W2)
    prob = tf.nn.softmax(logits, name='softmax')

    # Loss between taken action & distribution
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    immediate_rewards = tf.placeholder(dtype=tf.float32, shape=[None,1], name="immediate_rewards")

    # Function weighting our rewards
    discount_f = lambda accum, val: accum * GAMMA + val;
    total_rewards_reversed = tf.scan(discount_f, tf.reverse(immediate_rewards, [True, False]))
    total_rewards = tf.reverse(total_rewards_reversed, [True, False])

    # Normalize discounted_epr to standard mean and variance
    mean, variance = tf.nn.moments(total_rewards, [0], shift=None, name="total_reward_moments")
    total_rewards -= mean
    total_rewards /= tf.sqrt(variance + 1e-6)

    # Keep track of iters
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    # Optimizer
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    # Use normalized_rewards for loss gradient
    grads = optimizer.compute_gradients(loss, grad_loss=total_rewards)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)

    with tf.Session() as sess:
        # Initialize our variables
        sess.run(tf.global_variables_initializer())

        ## Restore checkpoint and write graph
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(f'out/{NAME}/checkpoints/')
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring from checkpoint")
            saver.restore(sess, ckpt.model_checkpoint_path)  
        writer = tf.summary.FileWriter(f'out/{NAME}/graphs/{NAME}', sess.graph)
        episode_number = global_step.eval()
        ##

        # Last image
        prev_x = None

        # Histories
        x_hist, reward_hist, y_hist = [],[],[]
        
        # Game vars
        running_reward = None
        reward_sum = 0
        
        while True:
            # Preprocess the observation
            cur_x = preprocess(observation)
            # Set input to different between current and last image
            x = cur_x - prev_x if prev_x is not None else np.zeros(IMAGE_SIZE)
            prev_x = cur_x

            # Get prob dist for given image
            feed = { X: np.reshape(x, (1,-1))  }
            sample_prob = sess.run(prob, feed)
            sample_prob = sample_prob[0,:]
            # Stochastically sample a policy
            action = np.random.choice(N_ACTIONS, p=sample_prob)
            label = np.zeros_like(sample_prob)
            label[action] = 1

            # Step the environment
            observation, reward, done, info = env.step(action+1)
            reward_sum += reward

            # Record game history
            x_hist.append(x)
            y_hist.append(label)
            reward_hist.append(reward)

            if done:
                # Update running reward
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

                # Save reward value for graping later
                dataSaver.add('reward', reward_sum, episode_number)

                # Update parameters
                feed = {X: np.vstack(x_hist),
                        immediate_rewards: np.vstack(reward_hist),
                        Y: np.vstack(y_hist)}
                _ = sess.run(train_op,feed)

                # Save checkpoint
                if episode_number % 100 == 0:
                    saver.save(sess, f'out/{NAME}/checkpoints/{NAME}')
                    dataSaver.save()

                # Print out progress
                if episode_number % 10 == 0:
                    print('ep {}: reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward))
                else:
                    print('\tep {}: reward: {}'.format(episode_number, reward_sum))

                # Reset game vars
                x_hist, reward_hist, y_hist = [],[],[] # reset game history
                observation = env.reset() # reset env
                reward_sum = 0
                episode_number += 1

def record(modelName='pong'):  
    from gym import wrappers
    env = gym.make('Pong-v0')
    env = wrappers.Monitor(env, f'out/{NAME}/recording', force=True)    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(f'out/{NAME}/checkpoints/{modelName}.meta')
        ckpt = tf.train.get_checkpoint_state(f'out/{NAME}/checkpoints/')
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring from checkpoint")
            saver.restore(sess, ckpt.model_checkpoint_path)  
        graph = tf.get_default_graph()
        global_step = graph.get_tensor_by_name('global_step:0')
        print("Recoding at episode {}".format(global_step.eval()))
        X = graph.get_tensor_by_name("X:0")
        prob = graph.get_tensor_by_name("softmax:0")

        observation = env.reset()
        prev_x = None
        while True:
            cur_x = preprocess(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros(IMAGE_SIZE)
            prev_x = cur_x
            feed = { X: np.reshape(x, (1,-1))  }
            sample_prob = sess.run(prob, feed)
            sample_prob = sample_prob[0,:]
            action = np.argmax(sample_prob)+1
            observation, _, done, _ = env.step(action)
            if done:
                break       

def graph():
    from matplotlib import pyplot as plt
    plt.title("100 Episode AI Reward Average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    dataSaver = DataSaver(NAME)
    dataSaver.graph('reward')

def main(args):
    if 'record' in args:
        record()
    elif 'graph' in args:
        graph()
    else:
        train()

if __name__ == '__main__':
    from sys import argv
    main(argv)