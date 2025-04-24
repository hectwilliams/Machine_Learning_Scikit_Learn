#! /usr/bin/env python3

import os 
import sys 
import subprocess
import numpy as np 
import gymnasium as gym
import tensorflow as tf 
from google.cloud import storage 
from pathlib import Path
from google.cloud.storage import Client, transfer_manager

BUCKET_NAME = 'train_bucket_0001'
STORAGE_OBJECT_NAME = "train_lunar_lander_model"
   
model_version = "0004"
model_name = "lunar_lander_model"
model_path = os.path.join(os.getcwd(), model_name, model_version)

def upload_directory_with_transfer_manager(bucket_name, source_directory, workers=8):
    """Upload every file in a directory, including all files in subdirectories.

    Each blob name is derived from the filename, not including the `directory`
    parameter itself. For complete control of the blob name for each file (and
    other aspects of individual blob metadata), use
    transfer_manager.upload_many() instead.
    """

    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The directory on your computer to upload. Files in the directory and its
    # subdirectories will be uploaded. An empty string means "the current
    # working directory".
    # source_directory=""

    # The maximum number of processes to use for the operation. The performance
    # impact of this value depends on the use case, but smaller files usually
    # benefit from a higher number of processes. Each additional process occupies
    # some CPU and memory resources until finished. Threads can be used instead
    # of processes by passing `worker_type=transfer_manager.THREAD`.
    # workers=8

    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)

    # Generate a list of paths (in string form) relative to the `directory`.
    # This can be done in a single list comprehension, but is expanded into
    # multiple lines here for clarity.

    # First, recursively get all files in `directory` as Path objects.
    directory_as_path_obj = Path(source_directory)
    paths = directory_as_path_obj.rglob("*")

    # Filter so the list only includes files, not directories themselves.
    file_paths = [path for path in paths if path.is_file()]

    # These paths are relative to the current working directory. Next, make them
    # relative to `directory`
    relative_paths = [path.relative_to(source_directory) for path in file_paths]

    # Finally, convert them all to strings.
    string_paths = [str(path) for path in relative_paths]

    print("Found {} files.".format(len(string_paths)))

    # Start the upload.
    results = transfer_manager.upload_many_from_filenames(
        bucket, string_paths, source_directory=source_directory, max_workers=workers
    )

    for name, result in zip(string_paths, results):
        # The results list is either `None` or an exception for each filename in
        # the input list, in order.

        if isinstance(result, Exception):
            print("Failed to upload {} due to exception: {}".format(name, result))
        else:
            print("Uploaded {} to {}.".format(name, bucket.name))

def exponential_decay(lr0:float, s:int):
    def exponential_decay_function(epoch):
        return lr0 * 0.1**(epoch/s)
    return exponential_decay_function

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
        return obs, reward, done, grads, loss         

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
    
    Note:
        obs
        [x coordinate, y coordinate, x velocity , y velocity , angle, angular velocity, left leg lands on ground, right leg lands on ground]

    """
    grads_per_episode = []
    rewards_per_episode = [] 
    loss_per_episode = [] 

    for episode in range(n_episodes):
        rewards = []
        grads = []
        losses = []
        obs, info = env.reset() 

        for step in range(n_max_steps):
            obs, reward, done, grad, loss = play_single_step(env, obs,model,loss_fn)

            # charge for moving away from 0,0 point
            reward += -np.tanh( np.linalg.norm(obs[:2] ,ord=2) )
            # reward += 1e-18 * -np.abs(np.tanh(obs[0:1]) ) # charge x for offset from origin

            #  charge velocity vector
            reward += -np.tanh(np.linalg.norm(obs[2:4]))
            reward += 1e-12 * -np.abs(np.tanh(obs[3:4]) ) # charge speed in vertical direction
            
            #  charge angular velocity 
            reward += -np.tanh(np.linalg.norm(obs[5:6]))

            rewards.append(reward)
            grads.append(grad)
            losses.append(loss)

            if done:
                break

        loss_per_episode .append(losses)
        rewards_per_episode.append(rewards)
        grads_per_episode.append(grads)

    return rewards_per_episode, grads_per_episode, loss_per_episode

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
    exponential_decay_func  = exponential_decay(0.04, 350)
    mean_reward_less_2_count = 0
    moving = np.zeros((10))

    for iteration in range(n_iterations):
        
        lr  = exponential_decay_func(iteration)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        rewards_per_episode, grads_per_episode, losses_per_episode = play_episodes(env, n_episodes,n_max_steps,model, loss_fn)
        discounted_normalized_rewards_per_episode = discount_and_normalize_rewards(rewards_per_episode,discount_factor)
        mean_grads = []
        captured_good_model_state = False 

        for trainable_variables_index in range(len(model.trainable_variables)):

            weighted_trainable_variables = [ discounted_normalized_reward * grads_per_episode[episode_index][step][trainable_variables_index] # grads[0][0][0] grads[1][0][0] grads[2][0][0]  ... grads[0][1][0] grads[1][1][0] grads[2][1][0] ... grads[0][0][1]  grads[1][0][1]  grads[2][0][1] ...
    
            for episode_index, discounted_normalized_rewards in enumerate(discounted_normalized_rewards_per_episode)
                for step, discounted_normalized_reward in enumerate(discounted_normalized_rewards) ]
            weighted_trainable_variables_mean = tf.reduce_mean(weighted_trainable_variables, axis = 0)
            mean_grads.append(weighted_trainable_variables_mean)

        optimizer.apply_gradients(zip(mean_grads, model.trainable_variables))
        sum_of_discounted_rewards[iteration] = np.sum(np.concatenate(discounted_normalized_rewards_per_episode))
        
        log_dict = {'iteration': str(iteration + 1) , 'lr': np.round(lr,4).__str__(), 'mean_reward': np.round(np.mean(np.concatenate(rewards_per_episode)),4).__str__() , 'mean_loss': np.mean(np.concatenate(losses_per_episode)).round(4).__str__()}

        mean_reward_less_2_count += float(log_dict['mean_reward']) > -2.0
        moving[iteration % 10] = float(log_dict['mean_loss'])
        captured_good_model_state = captured_good_model_state or (float(log_dict['mean_reward']) > -2.0)
        
        if (iteration >= 30 and mean_reward_less_2_count < 5) or ( iteration > 20 and (np.sum(moving)/10) < 0.9 ) or (iteration == 20 and not captured_good_model_state):  
            return False 
        
        if (iteration + 1) % 5  == 0:

            # save to working directory in locally (i.e. container)
            tf.saved_model.save(model, model_path)
            
            # upload model 
            upload_directory_with_transfer_manager(BUCKET_NAME, model_path) # https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python

            break 
    return True

def gen_model(random_seed_kernel):
    """
        Generates Tensorflow model (i.e. Policy)

    Args:

        random_seed_kernel - seed value determines sampling breadth used for tf.keras.initializers.LecunUniform
    
    Returns:

        Tensorflow Model 
    """
    n_inputs = 8 
    seed_ = tf.keras.random.SeedGenerator(seed=random_seed_kernel)
    data_in = tf.keras.layers.Input(shape=(n_inputs,))
    z = tf.keras.layers.Dense(32, activation='elu', kernel_initializer=tf.keras.initializers.LecunUniform(seed=seed_)) (data_in)
    z = tf.keras.layers.Dense(16, activation='elu', kernel_initializer=tf.keras.initializers.LecunUniform(seed=seed_)) (z)
    z = tf.keras.layers.Dense(8, activation='elu', kernel_initializer=tf.keras.initializers.LecunUniform(seed=seed_)) (z)
    data_out = tf.keras.layers.Dense(4, activation='softmax')(z)
    model = tf.keras.Model(inputs=[data_in], outputs=[data_out])
    print('KERNEL RANDOM SEED', random_seed_kernel)
    return model 


env_options = dict(id="LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode=None)

if __name__ == '__main__':

    env = gym.make(**env_options)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    obs, info = env.reset() 
    discount_factor = 0.90
    n_iteration = 500
    n_episodes = 10                                                                                                                                                                                                                                                                                                                       
    n_steps  = 500 
    num_of_sync_runs = 0
    seed = 766078
    model = gen_model(seed)    

    for i in range(1):
        ret = train_policy(env, n_iteration, n_episodes, n_steps, model, loss_fn, optimizer, discount_factor)
        if ret:
            break
        else:
            seed = int(tf.random.uniform((),minval=0, maxval=1000000, dtype=tf.int32, seed=42).numpy())
            model = gen_model(seed)
