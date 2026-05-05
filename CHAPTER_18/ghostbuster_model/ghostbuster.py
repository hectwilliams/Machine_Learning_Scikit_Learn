import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
import os
import PIL
import tensorflow as tf
from collections import deque
import pandas as pd
import pickle
from collections import defaultdict
import csv

PREFIX = '/content/drive/MyDrive/Colab Notebooks/rlearning/test_01/'
MARKER_PT_SIZE = 10
IMAGES_DIR = "/content/"
SIDE_LENGTH = 1000
GHOST_SIZE = 100
DUMMY_IMG = np.ones((GHOST_SIZE, GHOST_SIZE)) * 0.5
BATCH_SIZE = 32
DISCOUNT_RATE = 0.95
LEARNING_RATE = 1e-3
ZETA_HYPERPARAMETER = 0 # control importance sampling focus (zeta = 0 {uniform sampling}, zeta = 1{importance sampling})
BETA_HYPERPARAMETER = 0 # (0 - no compensation)  (1 - compensation) # compensate importance sampling
TINY_FLOAT = 1e-10
TRAINING = 0
MAX_ACTIONS_PER_EPISODE = 100
MAX_NUMBER_OF_GHOSTS = 4
NUM_CELLS = 3
N_EPISODES_TRAIN_THRESHOLD = 10000 
N_EPISODES = 30000 
N_TEST_EPISODES = 300
DF_SIZE = 45000
TARGET_MODEL_FILENAME = "target_model.weights.h5"
MODEL_FILENAME = "model.weights.h5"
LOG_FILENAME = "log_inference_results.csv"

fig, ax = plt.subplots()
tf.random.set_seed(1)
np.random.seed(1)

def sq_inits(number):

  '''
    computes origin points and loads to array. The number of init points are
    controlled by number argument

    _ __ _ _ _ _
    |            |
    |            |
    |     A      |
    |            |
    |._ _ _ _ _ _|
    0,0 is A cell


      _ __ _ _ _ __ _ _ _ _ _
    |            |           |
    |            |           |
    |     A      |    B      |
    |            |           |
    |._ _ _ _ _ _|._ _ _ _ _ |
    |            |           |
    |            |           |
    |            |     C     |
    |            |           |
    |__ _ _ _ _ _|._ __ _ _ _|

    1,0 in A cell
    0,1 in B cell
    1,1 in C cell

  '''

  if number <=0:
    raise ValueError("The provided value is invalid.")

  ret = []

  for f in range(0,number - 1):
  # add row of cells (top right cell not included)
    ret.append( Square( number - 1, f ) )
  # add column of cells (top right cell not included)
    ret.append( Square( f, number - 1 ) )
  # add cell in top right section
  ret.append( Square(number-1,number-1))

  return ret

def show_board(board):
  '''

  Display board on terminal. The following identifiers represent the state of
  a cell:

  [] - empty cell
  ]Q] - ghost
  [!Q] - consumed ghost

  '''
  print("__")
  for r in range(board.n_cells-1,-1, -1): # ( descending order for row axis)
    for c in range(board.n_cells ): # ascending order for column axis
      ch = ""
      if board.board_[r][c]  & 1 << 0:
        ch = "!"+ ch
      else:
        ch = " " + ch

      if board.board_[r][c] & 1 << 1:
        ch = "Q" + ch
      else:
        ch = " " + ch

      ch = "[" + ch + "]"
      print(f"{ch}",end="\t")
    print()
  print("__")

def calc_extent(board, r, c):

  """
    Returns extent of ghostbuster image.

    Param:
      board - Board object
      r - row of ghostbuster
      c - column of ghostbuster
  """

  mid_r = ( r + (r + SIDE_LENGTH) ) //2 # (pos1 + pos2) / 2
  mid_c = ( c + (c + SIDE_LENGTH) ) //2

  x_gb_r_min = mid_r - GHOST_SIZE # GHOST_SIZE//2
  x_gb_r_max = mid_r + GHOST_SIZE # GHOST_SIZE//2

  x_gb_c_min = mid_c - GHOST_SIZE # GHOST_SIZE//2
  x_gb_c_max = mid_c + GHOST_SIZE # GHOST_SIZE//2

  return x_gb_c_min,x_gb_c_max, x_gb_r_min, x_gb_r_max

class Square:

  def __init__(self, r = 0, c = 0):

    self.r = r * SIDE_LENGTH
    self.c = c * SIDE_LENGTH

    # vertices points (row, column)
    self.v_topLeft = (self.r+SIDE_LENGTH, self.c)
    self.v_topright = (self.r + SIDE_LENGTH,self.c + SIDE_LENGTH) # (self.r + 1,self.c + 1)
    self.v_bottomLeft = (self.r ,self.c)
    self.v_bottomRight = (self.r ,self.c + SIDE_LENGTH)

    self.lines =[
        [ self.v_topLeft, self.v_topright ],
        [ self.v_topright, self.v_bottomRight ],
        [ self.v_bottomRight, self.v_bottomLeft  ] ,
        [ self.v_bottomLeft, self.v_topLeft  ]
    ]

    self.center = (self.r + SIDE_LENGTH*0.5, self.c + SIDE_LENGTH*0.5)

    self.players = np.zeros(shape=(3)) # - 0-ghost 1-pacman 2-food

    self.s_ghost = None # ghost img object
    self.s_ghostbuster = None # ghost img object
    self.s_ghostbuster_prev_pos = None # ghost img object

class Board:

  def __init__(self, n_cells = 1):

    # axis
    ax.set_xlim(0, SIDE_LENGTH*n_cells)
    ax.set_ylim(0, SIDE_LENGTH*n_cells)

    self.n_cells = n_cells
    self.board_ = np.zeros(shape=(n_cells, n_cells), dtype=np.int32)
    self.unique_visits = {}
    self.draw_lock = False
    self.ghost_inited = False
    self.ghostbuster_inited = False
    self.n_ghosts = 0
    self.ghosts_eaten = 0
    self.loopArr = [ "", "", "", "", "" ]
    self.loopArrIndex = 0
    self.state_action = [ "", "" ]
    self.state_action_index = 0

    if (n_cells < 1):
      raise ValueError("The provided value is invalid.")

    self.squares = []

    for k in range(1, n_cells + 1 ):
      self.squares += sq_inits(k)

    self.ghost_init()

    self.ghostbuster_init()

    # self.draw_repr_board()
    # print()

  def draw_repr_board(self):
    """
      Draws board representation with ghost.
    """
    if  self.draw_lock:
      return

    for  sq in self.squares:

      ax.scatter(*sq.v_bottomLeft, MARKER_PT_SIZE, color='red')

      ax.scatter(*sq.v_bottomRight, MARKER_PT_SIZE, color = "red")

      ax.scatter(*sq.v_topLeft, MARKER_PT_SIZE, color = "red")

      ax.scatter(*sq.v_topright, MARKER_PT_SIZE, color = "red")

      for line in sq.lines:
        ax.plot( *zip(*line), color = "blue", linewidth=1, linestyle= "--")

      # add random ghost to board
    self.draw_lock = True


  def ghost_init(self):

    '''
    Randomly inserts ghosts onto board. The function loops 15 times and may or may not insert ghost at random location.

    '''
    if  self.ghost_inited:
      return

    n_loops =  MAX_NUMBER_OF_GHOSTS # max possible number of ghosts residing on board

    indices = np.random.randint(0,len(self.squares), size=((MAX_NUMBER_OF_GHOSTS)))

    for some_index in (indices):

      sq = self.squares[some_index] # random square

      if not self.board_[sq.r//SIDE_LENGTH][sq.c//SIDE_LENGTH] & 1<< 1: # prevents repeating count of ghosts
        self.n_ghosts += 1

      self.board_[sq.r//SIDE_LENGTH][sq.c//SIDE_LENGTH] = self.board_[sq.r//SIDE_LENGTH][sq.c//SIDE_LENGTH]  | 1 << 1

      if (not sq.s_ghost ):

        img_path = PREFIX + "ghost.png"
        img = PIL.Image.open(img_path)
        img_resized = img.resize((GHOST_SIZE, GHOST_SIZE), PIL.Image.Resampling.LANCZOS)

        # small
        mid_r = ( sq.r + (sq.r + SIDE_LENGTH) ) //2 # (pos1 + pos2) / 2
        mid_c = ( sq.c + (sq.c + SIDE_LENGTH) ) //2

        x_ghost_r_min = mid_r - GHOST_SIZE # GHOST_SIZE//2
        x_ghost_r_max = mid_r + GHOST_SIZE # GHOST_SIZE//2

        x_ghost_c_min = mid_c - GHOST_SIZE # GHOST_SIZE//2
        x_ghost_c_max = mid_c + GHOST_SIZE # GHOST_SIZE//2

        self.s_ghost  = ax.imshow(img_resized, extent = ( x_ghost_c_min, x_ghost_c_max, x_ghost_r_min, x_ghost_r_max)  )  # extent=[x_min, x_max, y_min, y_max]

    self.ghost_inited = True

  def ghostbuster_init(self):

    '''
      Adds ghostbuster to board. Ghostbuster is added to origin at the bottom left of board.
    '''

    if self.ghostbuster_inited :
      return

    sq_origin = self.squares[0]

    self.board_[sq_origin.r][sq_origin.c] =  self.board_[sq_origin.r][sq_origin.c] |  1<<0

    self.s_ghostbuster_prev_pos = (sq_origin.r, sq_origin.c) # store location

    self.s_ghostbuster  = ax.imshow(DUMMY_IMG, extent = calc_extent(self, sq_origin.r, sq_origin.c), cmap="Greys", vmin=0, vmax=1, alpha=0.2)

    self.ghostbuster_inited = True

    self.unique_visits[sq_origin.r * self.n_cells + sq_origin.c] = 1 # set ghostbuster

  def get_square(self, r, c):

    for sq in self.squares:
      if sq.r == r and sq.c == c:
        return sq

  def get_state(self):
    '''
      Returns state of board. The state is represented by a n_cells^2 array.
    '''
    n_dimension = 2

    ret = np.zeros(shape=(self.n_cells * self.n_cells + n_dimension), dtype=np.int32)

    # grid
    for r in range(self.n_cells):
      for c in range(self.n_cells):
        if self.board_[r][c] & 1 << 1:
          ret[r * self.n_cells + c] = 1

    # position stored in last two array slots
    ret[-2] = self.s_ghostbuster_prev_pos[0] # row
    ret[-1] = self.s_ghostbuster_prev_pos[1] # col

    # include number of eaten

    ret = np.hstack((ret, [self.ghosts_eaten]))
    return ret

  def get_state_to_dec(self, state):
    """
    Returns state of board in decimal representation. The board state excluding ghostbuster needs all state values but the last 2

    """
    ret = 0
    s = state[:-2]
    length = len(s)
    for i in range(len(s)):
      ret = ret  | (  s[i]    << length - 1 - i)
    return ret

  def play(self, state, model,target_model, replay_buffer,df, epsilon, table_Q):

    done = False
    dir = ["left", "right", "up", "down"]

    rr = self.s_ghostbuster_prev_pos[0]
    cc = self.s_ghostbuster_prev_pos[1]
    rrr = rr
    ccc = cc
    prev_ate = self.ghosts_eaten
    found_ghost = False

    # check if ghostbuster eats ghost at current state

    if self.board_[rr][cc] & 1 << 1:
      # remove ghost
      self.board_[rr][cc] = self.board_[rr][cc] ^ (1 << 1)
      self.ghosts_eaten +=1
      found_ghost = 1
      some_square = self.get_square(rr, cc)

      if  some_square:
        if some_square.s_ghost: # ghost present in cell
          some_square.s_ghost.remove()
          some_square.s_ghost = None

    d_action = epislon_greedy_policy(state, model, epsilon) # action


    action = [
      'left', # 0
      'right', # 1
      'up', # 2
      'down' # 3
    ][d_action]

    table_Q[f"(state)(action)"] = d_action

    # potential ghostbuster move test
    if action == "left" and cc > 0:
      ccc -= 1

    if action == "right" and (cc < self.n_cells - 1):
      ccc += 1

    if action == "up" and (rr < self.n_cells - 1):
      rrr += 1

    if action == "down" and rr > 0:
      rrr -= 1

    # move ghostbuster
    if rrr != rr or ccc != cc:

      # remove ghostbuster from current(i.e. previous) pos
      self.board_[rr][cc] = self.board_[rr][cc] ^ (1 << 0)

      # move variables ghostbuster
      rr = rrr
      cc = ccc
      self.s_ghostbuster_prev_pos = (rr, cc)

      # set/move ghostbusttghjer position
      self.board_[rr][cc] = self.board_[rr][cc] | (1 << 0)

      # remove thumbnail of ghostbuster
      self.s_ghostbuster.remove()
      self.s_ghostbuster = None

      # set/move thumbnail of ghostbuster
      self.s_ghostbuster  = ax.imshow(DUMMY_IMG, extent = calc_extent(self, rr, cc), cmap="Greys", vmin=0, vmax=1)

      next_state = self.get_state() # number of ghosts ate are updated in last array slot

      transition =  f"{state}-{action}-{next_state}"

      self.loopArr[self.loopArrIndex] = transition

      self.loopArrIndex = (self.loopArrIndex + 1) % len(self.loopArr)

      # loop check
      if self.state_action[ self.state_action_index ] == transition  :

        record = (state, d_action, -100 , next_state, True)

        df.loc[len(df.index)] = [record, 1, transition]

        return next_state, -100 , True, d_action, state

      self.state_action [(self.state_action_index + 1) % 2] = transition

      self.state_action_index = (self.state_action_index + 1) % 2

      # check if loop

      self.unique_visits[rr * self.n_cells + cc] = 1

      rward = len(self.unique_visits ) + self.ghosts_eaten # number of unique steps by ghostbuster + number of ghosts removed by ghostbuster

      record = (state, d_action, rward , next_state, self.ghosts_eaten == self.n_ghosts)

      if len(df) >= DF_SIZE:
        #remove top 5K experiences if experiences storage reaches 40000
        df = df.iloc[15000:].reset_index(drop=True)

      df.loc[len(df.index)] = [record, 1, transition]

      return next_state, rward , self.ghosts_eaten == self.n_ghosts, d_action, state

    else:

      #unsafe move near boundary ( noop )

      done = True

      transition =  f"{state}-{action}-{state}"

      rward = -1 # len(self.unique_visits ) + self.ghosts_eaten

      record = (state, d_action, rward, state, done)

      if len(df) >= DF_SIZE:
        #remove top 5K experiences if experiences storage reaches 40000
        df = df.iloc[15000:].reset_index(drop=True)

      df.loc[len(df.index)] = [record, 1, transition]

      return  state, rward, done, d_action, state # state, reward, done

def epislon_greedy_policy(state, model, epsilon=0):
  '''
    Choose random action(1,2,3, or 4) using episolon greedy policy.

    Param:

      state - flatten array with all but last two values are 1 or 0 indicating
      whether ghost is present. The remaining array slots are the position of
      the ghostbuster on 2D grid

      model - tensorflow model

      epsilon - hyperparameter controlling how the training model chooses actions

  '''

  if np.random.rand() < epsilon:
    action = np.random.randint(4)
  else:
    s = state[np.newaxis]
    s = tf.convert_to_tensor(s)
    states_in0, states_in1, states_in2, states_in3 = s[:,:9], s[:,9:10], s[:,10:11], s[:, 11:12]
    Q_values = model( (states_in0, states_in1, states_in2, states_in3), verbose=0 ) # Q_values dimen = [1,4]
    action = np.argmax(Q_values[0])   # optimal Q(state', action')

  return action

def get_state_to_dec(state):
  """
  Returns state of board in decimal representation. The board state excluding ghostbuster needs all state values but the last 2

  """
  length = tf.shape(state)[-1]
  num_ele = tf.shape(state)[0]
  ret2 = tf.Variable(tf.zeros(shape=(num_ele,1), dtype=tf.int32))

  for j in range(num_ele):
    state_o = state[ j : j+1]
    for i in range(length):
      ret2[j, 0].assign( ret2[j, 0] + state_o[0, i] * 2**(length - 1 - i)   )
  ret2 = tf.convert_to_tensor(ret2)
  return ret2

def create_model(n_inputs, n_outputs, func, embedding1, embedding2, embedding3, embedding4):
  '''
    Creates a tensorflow model.

    Param:

      n_inputs - number of input neurons

      n_outputs - number of output neurons

  '''

  state_ = tf.keras.layers.Input(shape=(n_inputs - 2,) )
  position_row = tf.keras.layers.Input(shape=(1,))
  position_col = tf.keras.layers.Input(shape=(1,))
  ghosts = tf.keras.layers.Input(shape=(1,))

  #state of environment
  z_state = tf.keras.layers.Lambda(lambda x: func(x) , output_shape=(1,) , dtype=tf.int32 )(state_)
  z_state = embedding1(z_state)
  z_state = tf.keras.layers.Flatten()(z_state)

  #row position
  hidden_0_row = embedding2(position_row)
  hidden_0_row = tf.keras.layers.Flatten()(hidden_0_row)

  #column position
  hidden_0_col = embedding3(position_col)
  hidden_0_col = tf.keras.layers.Flatten()(hidden_0_col)

  #number of ghosts in environment
  z_ghost = embedding4(ghosts)
  z_ghost = tf.keras.layers.Flatten()(z_ghost)

  z = concat = tf.keras.layers.concatenate([z_state, hidden_0_row, hidden_0_col, z_ghost])

  for i in range(4):
    z = tf.keras.layers.Dense(64,  activation="selu", kernel_initializer= tf.keras.initializers.GlorotNormal(seed=1) ) (z)

  output = tf.keras.layers.Dense(n_outputs )(z)

  model = tf.keras.Model(inputs=(state_, position_row, position_col, ghosts), outputs=[output])

  return model

def predict_next_state_action(rewards, actions, dones, states, next_states, model, target_model):

  '''
    Predict action using model.

    Param:
      state - A batch element. A flattened array with all but last two values are 1 or 0 indicating
      whether ghost is present. The remaining array slots are the position of
      the ghostbuster on 2D grid

      model - tensorflow model
  '''

  # next action
  next_states = tf.convert_to_tensor(next_states)

  # input to model
  next_states_in0, next_states_in1 , next_states_in2, next_states_in3 = next_states[:,:9], next_states[:,9:10] , next_states[:,10:11], next_states[:,11:12]

  # Estimate Q(s', ?')
  next_Q_values = target_model ( (next_states_in0, next_states_in1, next_states_in2, next_states_in3) ,verbose=0) # online model

  # Max Estimate Q(s', a') (optimal --> ? = a')
  max_next_Q_values = tf.reduce_max(next_Q_values, axis=1, keepdims=True)

  reward_tf = tf.constant(rewards, dtype=tf.float32)[:, None]
  done_tf = (tf.constant(dones, dtype=tf.float32)[:, None])
  actions_tf = tf.constant(actions, dtype=tf.float32)[:, None]

  target_Q_values = ( reward_tf + (1 - done_tf ) * DISCOUNT_RATE * max_next_Q_values)


  return target_Q_values, actions_tf

def samples_experience(replay_buffer, df, beta):

  '''
    Samples experience from replay buffer.

    Param:

      replay_buffer - deque of experiences

      df - dataframe with experiences and priority features

      beta - hyperparameter controlling importance of priority

    Returns:

      states, actions, rewards, next_states, dones, dataframe ids - batch of experiences
  '''
  sampled_df = df.sample(n=len(df) if len(df) < BATCH_SIZE else BATCH_SIZE, weights='Priority')

  df_exp =  sampled_df["Experiences"]

  df_exp_index = df_exp.index.tolist()

  df_list = df_exp.to_list()

  states, actions, rewards, next_states, dones   = [

    np.array([experience[index] for experience in df_list])

    for index in range(5)

  ]

  return states, actions, rewards, next_states, dones, df_exp_index #, weight_compensate


def train_step(replay_buffer, df, model, target_model, beta, optimizer, loss_fn):
  '''
    Train model per step

    Param:

      replay_buffer - deque of experiences

      df - dataframe with experiences and priority features

      model - tensorflow model

      target_model - tensorflow model

      beta - hyperparameter controlling importance of priority

    Returns:

      loss - loss value
  '''
  experiences = samples_experience(replay_buffer, df, beta)
  states, actions, rewards, next_states, dones, df_ids = experiences

  target_Q_values, actions_mask = predict_next_state_action(rewards, actions, dones, states, next_states, model, target_model)

  with tf.GradientTape() as tape:

    s = tf.convert_to_tensor(states)

    # input to model
    next_states_in0, next_states_in1, next_states_in2, next_states_in3 = s[:, :9], s[:, 9:10], s[:, 10:11], s[:, 11:12]

    # Estimate Q(s, ?)
    all_Q_values = model( (next_states_in0, next_states_in1, next_states_in2, next_states_in3) , verbose=0) # online model

    # Select action for state using optimal Next State Q-Value ( think of this as a look ahead to better future outcomes )
    Q_values_predict = tf.reduce_sum(all_Q_values * actions_mask, axis=1, keepdims=True)

    loss = tf.reduce_mean( loss_fn(target_Q_values, Q_values_predict) ) # sample_weights

    ## prioritized learning not used ! 

    # temporal_diff_err_Priority = tf.abs(tf.subtract(target_Q_values, Q_values_predict))

    # temporal_diff_err_Priority = temporal_diff_err_Priority **ZETA_HYPERPARAMETER # probability

    # weight_compensate = 1 / ( (len(df) * temporal_diff_err_Priority)**beta )

    # temporal_diff_err_Priority = temporal_diff_err_Priority * weight_compensate

    # df.loc[df_ids, 'Priority'] = temporal_diff_err_Priority.numpy().tolist()

    # Tally occurrences experiences (state-action-nextstate)

    # for i in range(len(df_ids)):

    #   ss = str(s[i].numpy())
    #   nn = str(nxt[i].numpy())
    #   aa = str(a[i].numpy())
    #   Q_table[ f"{ ss }, {aa}, {nn} " ]  += 1


  # gradient descent
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return loss

def num_ghosts(board):
  """
    Returns the number of ghosts in the board.
  """
  ret = 0
  for r in range(board.n_cells):
    for c in range(board.n_cells):
      # print(r,c,board.board_[r][c])
      ret += int(board.board_[r][c] >=2) # ghosts are represented by bit 1 of value in cell(i.e. square)
  return ret

n_cells = NUM_CELLS
n_cells_sq = n_cells * n_cells
n_dim = 2
n_outputs = 4
n_inputs = n_cells_sq + n_dim
total_actions_per_game = MAX_ACTIONS_PER_EPISODE #(n_cells * n_cells) **2
training = TRAINING
n_episodes = N_EPISODES if training else N_TEST_EPISODES
target_model = None
model = None
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
embedding1 = tf.keras.layers.Embedding(2**(n_cells_sq), 3)
embedding2 = tf.keras.layers.Embedding(20, 2)
embedding3 = tf.keras.layers.Embedding(20, 2)
embedding4 = tf.keras.layers.Embedding(5, 2)
model = create_model(n_inputs, n_outputs, get_state_to_dec, embedding1, embedding2, embedding3,embedding4)
loss_f = tf.keras.losses.MeanAbsoluteError()
# loss_f = tf.keras.losses.SparseCategoricalCrossentropy()
score_metric = np.zeros(shape=(N_TEST_EPISODES))

target_model = tf.keras.models.clone_model(model)
target_model.set_weights(model.get_weights())

if os.path.exists(PREFIX + TARGET_MODEL_FILENAME):
  target_model.load_weights(PREFIX + TARGET_MODEL_FILENAME)
  model.load_weights(PREFIX + MODEL_FILENAME)
  with open(PREFIX + 'opt.pkl', 'rb') as file:
    record = pickle.load(file)
    optimizer = record.get("opt")
  print("previous model weights loaded")

model.compile()
target_model.compile()

print(model.summary())
print(target_model.summary())

replay_buffer = deque(maxlen=8000)
sum_of_rewards = np.zeros(shape=(n_episodes))
df = pd.DataFrame(columns=['Experiences', 'Priority', 'Transition'])
loss = loss_ = 2**32
histo_actions = defaultdict(int)

Q_table = defaultdict(int)


for episode in range(n_episodes):

  # reset game (new game)
  board = Board(n_cells=n_cells);
  state = board.get_state()
  acc_reward = 0
  print("start")

  # play game
  for index, step in enumerate(range(total_actions_per_game)):
    epsilon = max(1 - episode/N_EPISODES, 0.01) if training else 0
    state, reward, done, action_id, prev_state = board.play(state, model,target_model, replay_buffer, df, epsilon, Q_table) # batch of 32

    if reward > acc_reward:
      acc_reward = reward

    # gane over
    if done:
      show_board(board)
      print("end")
      break
    else:
      show_board(board)
    

  if training:

    # store reward accumulation per episode(i.e. game session)
    sum_of_rewards[episode]= acc_reward

    # train network
    if episode > N_EPISODES_TRAIN_THRESHOLD:

      beta = 0 if not BETA_HYPERPARAMETER else BETA_HYPERPARAMETER  + (1 - BETA_HYPERPARAMETER) * (episode/n_episodes)

      loss = train_step(replay_buffer, df,model,target_model, beta, optimizer, loss_f) # train step

      # update target weights
      if episode % 50 == 0:
        target_model.set_weights(model.get_weights())

    # save model
      if loss < loss_:
        loss_ = loss;
        print(f"reduce -- episode({episode})\tloss: {loss_}" )
        model.save_weights(PREFIX + MODEL_FILENAME)
        target_model.save_weights(PREFIX + TARGET_MODEL_FILENAME)
        with open(PREFIX +  'opt.pkl', 'wb') as file:
          pickle.dump({ "opt": optimizer}, file)
        # print(f"episode({episode})\tloss: {loss_}" )

    if (episode % 5000 == 0):
      print(f"Episode: {episode}\tLoss: {loss}\tReward_Median: {np.median(sum_of_rewards) ,} BUFFER_SIZE:{len(df)} " )

  else:
    # At this point we are running tests challenging model to different games
    score_metric[episode] += num_ghosts(board)
    a = num_ghosts(board)
    if episode == N_TEST_EPISODES - 1:
      print( f"average removed queeens: {np.mean(score_metric)}") # higher the better ()

if training:

  for key in Q_table:
    print(key, Q_table[key])

  plt.figure()
  plt.xlabel("Episode")
  plt.ylabel("Sum of Rewards")
  plt.plot(sum_of_rewards)



