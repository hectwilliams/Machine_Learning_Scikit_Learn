#! /usr/bin/env python3

import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
import os 
import sys 
MODEL_FILENAME = 'lunar_lander_model.keras'
SYNTH_DURATION = 1 # second
SYNTH_FREQ = 440

"""


Actions 
    0 - do nothing
    1 - fire left orientation engine 
    2 - fire main engine 
    3 - fire right orientation engine 

Observation 
    [
        x coordinate 

        y coordinate 

        x linear velocity

        y linear velocity 

        angle 

        angular velocity 

        left leg touch floor 

        right leg touch floor 
    ]

"""
import gymnasium as gym

def verbose_obs(obs):
    return dict(coord_xy=obs[:2], velocityXY=xy_to_phasor(obs[2:4]), angle=np.rad2deg(obs[5]) , left_leg_touch=obs[6], right_leg_touch=obs[7])

def xy_to_phasor(xy):
   return dict(mag=np.linalg.norm(xy, ord=2), angle=np.rad2deg(np.arctan(xy[1]/xy[0]))) 

def play_single_step(env, obs, model, loss_fn):
    """
    Agent takes an action based on current state of policy at time step in game

    
    Args:

        env - simulation environment

        obs - state of playable object 

        model - policy  

        loss_fn - categorical crossentropy loss 


    Returns:

        tuple with the following:

            obs - state of playable object after taking action  

            reward - loss/gain after taking policy's estimated action 

            done - flag indiciating if episode(i.e. game) is finished 

            grad - model gradient from estimation

    """
    with tf.GradientTape() as Tape:
        action_probabilities = model(obs[np.newaxis])
        action =  tf.argmax(action_probabilities, axis=1)
        target = tf.one_hot(action, depth=action_probabilities.shape[-1], axis=1)
        loss = loss_fn(target, action_probabilities)
        loss_mean = tf.reduce_mean(loss) # mean of batch
        grads = Tape.gradient(loss_mean, model.trainable_variables) 
        obs, reward, done, trunc, info  = env.step(action[0].numpy())
        return obs, reward, done, grads         

def play_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    """
    Play the the game n number of times.

    Args:

        env - simulation environment 

        n_episodes - number of games played

        n_max_steps - max number of time steps per game session

        model - policy 

        loss_fn - categorical cronssentropy loss

    Returns:

        tuple containing the following:
            rewards_per_episode - indexed by episode, each element is a list of rewards per time step
            grads_per_episode - indexed by episode, each element is a list of gradients per time step 
    """
    grads_per_episode = []
    rewards_per_episode = [] 
    for episode in range(n_episodes):
        # print( f'episode\t {episode}')
        rewards = []
        grads = []
        obs, info = env.reset() 
        for step in range(n_max_steps):
            obs, reward, done, grad = play_single_step(env, obs,model,loss_fn)
            rewards.append(reward)
            # print(f'avg reward:\t{np.mean(np.array(reward))}')
            grads.append(grad)
            if done:
                break 
        rewards_per_episode.append(rewards)
        grads_per_episode.append(grads)
    return rewards_per_episode, grads_per_episode

def discount_rewards(rewards: list, discount_factor: float):
    """
    Discount rewards 

    Args:

        rewards - list of rewards 

        discount_factor - controls whether we care for immediate rewards or future rewards 

    Returns:

        Discounted rewards of type np.array
    
    """
    discounted = np.array(rewards)
    for step in range(len(discounted)-2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted

def discount_and_normalize_rewards(rewards_per_episode, discount_factor) :
    """
    Discounts and standarize rewards. 

    Args:

        rewards_per_episode

        grads_per_episode

        discount_factor

    
    Returns:

        list of discounted-normalized rewards (rewards are numpy arrays )

    """
    discounted_rewards_per_episode = [discount_rewards(rewards, discount_factor) for rewards in rewards_per_episode] # returns list of numpy arrays 
    flatten_discount_rewards_per_episode = np.concatenate(discounted_rewards_per_episode) # [ np_array([1,2]) , np_array([3,4])] =  np.array([1,2,3,4])
    discounted_rewards_mean = flatten_discount_rewards_per_episode.mean()
    discounted_rewards_std = flatten_discount_rewards_per_episode.std()
    
    return [ (discounted_rewards - discounted_rewards_mean) / discounted_rewards_std  for discounted_rewards in discounted_rewards_per_episode] # standardize each np.array() in discounted_rewards_per_episode

def train_policy(env, n_iterations, n_episodes, n_max_steps, model, loss_fn, optimizer, discount_factor):
    """
    Train policy Neural Network 

    Args:

        env - environment

        n_iterations - number of iterations 

        n_episodes - number of episodes 

        n_max_steps - max number of steps per game 

        model - policy (i.e. neural network)

        loss_fn - loss function 

        optimizer - SGD

        discount_factor - 

    Returns:

        None 

    """
    
    sum_of_discounted_rewards = np.zeros(shape=(n_iterations))
    for iteration in range(n_iterations):
        rewards_per_episode, grads_per_episode = play_episodes(env, n_episodes,n_max_steps,model, loss_fn)
        discounted_normalized_rewards_per_episode = discount_and_normalize_rewards(rewards_per_episode,discount_factor)
        mean_grads = []
        
        for trainable_variables_index in range(len(model.trainable_variables)):

            weighted_trainable_variables = [ discounted_normalized_reward * grads_per_episode[episode_index][step][trainable_variables_index] # grads[0][0][0] grads[1][0][0] grads[2][0][0]  ... grads[0][1][0] grads[1][1][0] grads[2][1][0] ... grads[0][0][1]  grads[1][0][1]  grads[2][0][1] ...
    
            for episode_index, discounted_normalized_rewards in enumerate(discounted_normalized_rewards_per_episode)
                for step, discounted_normalized_reward in enumerate(discounted_normalized_rewards) ]
            weighted_trainable_variables_mean = tf.reduce_mean(weighted_trainable_variables, axis = 0)
            mean_grads.append(weighted_trainable_variables_mean)

        optimizer.apply_gradients(zip(mean_grads, model.trainable_variables))
        sum_of_discounted_rewards[iteration] = np.sum(np.concatenate(discounted_normalized_rewards_per_episode))
        model.save(os.path.join(os.getcwd(), MODEL_FILENAME))
        print(f"\r {iteration+1}/{n_iterations}")
    plt.plot(np.arange(n_iterations), sum_of_discounted_rewards)
    plt.xlabel('iteration')
    plt.ylabel('sum transformed rewards ')
    plt.show()


env_options = dict(id="LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5)
n_inputs = 8 
# sample_rate = 44100
# # generate a 1-second sine tone at 440 Hz
# y = np.sin(2 * np.pi * 440.0 * np.arange(sample_rate * 1.0) / sample_rate)
# tfm = sox.Transformer()
# tfm.build_file(
#     input_array=y, sample_rate_in=sample_rate,
#     output_filepath=os.path.join(os.getcwd(), 'debug_beep.wav' )
# )
# os.system('sox {} -qd'.format({os.path.join(os.getcwd(), 'debug_beep.wav' )}))
# assert(False)

if len(sys.argv) < 2:
    
    assert(False)

elif sys.argv[1] == 'train':
    
    env = gym.make(**env_options)
    input_ = tf.keras.layers.Input(shape=(n_inputs,))
    z = tf.keras.layers.Dense(128) (input_)
    z = tf.keras.layers.Dense(64) (z)
    z = tf.keras.layers.Dense(32) (z)
    z = tf.keras.layers.Dense(16) (z)
    z = tf.keras.layers.Dense(8) (z)
    output_ = tf.keras.layers.Dense(4, activation='elu')(z)
    model = tf.keras.Model(inputs=[input_], outputs=[output_])
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    # model.compile(loss=loss_fn, optimizer=optimizer)
    # replay_buffer = deque(maxlen=2000)
    obs, info = env.reset() 
    discount_factor = 0.90
    n_iteration = 2000
    n_episodes = 6
    n_steps  = 100 
    
    if os.path.exists(os.path.join(os.getcwd(), MODEL_FILENAME)):
        model = tf.keras.models.load_model(os.path.join(os.getcwd(), MODEL_FILENAME))
        
    train_policy(env, n_iteration, n_episodes, n_steps, model, loss_fn, optimizer, discount_factor)

elif sys.argv[1] == 'play':

    model = tf.keras.models.load_model(os.path.join(os.getcwd(), MODEL_FILENAME))
    env = gym.make(**{**env_options, 'render_mode':'human'})
    steps = 200
    episodes = 10

    for episode in range(episodes):
        obs, info = env.reset() 
        # print(env.action_space.sample())
        for i in range(steps):
            action_probabilities = model(obs[np.newaxis])
            action =  tf.argmax(action_probabilities, axis=1)
            action = action[0].numpy()
            random_sample = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)
            print(f'action: {action}')
            if done:
                print('--done--')
                break