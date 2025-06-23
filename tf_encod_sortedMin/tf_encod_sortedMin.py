# 2d_action_space_top_view_and_stack_height_isSorted
import pickle
import sys
import os
from typing import Dict

import tree
from tqdm import trange

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import time
import numpy as np
import pandas as pd
import sonnet as snt
import random
from datetime import datetime,timedelta
from collections import deque

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import tensorflow as tf

print("GPUs:", tf.config.list_physical_devices('GPU'))


# yard parameters
INITIAL_YARD_OCCUPIED_RATIO = 0.
MAX_DWELL_DAYS = 10 # 20
BAYS = 4  # X-axis
ROWS = 3  # Y-axis
TIERS = 3  # Stack height
FILL_RATIO = 0.9
NUM_CONTAINERS_PER_EPISODE = round(BAYS * ROWS * TIERS * (FILL_RATIO - INITIAL_YARD_OCCUPIED_RATIO))
# NUM_CONTAINERS_PER_EPISODE = 100


FILL_TIER0_AS_INITIALIZATION = False # how to initialize the yard block

# rewards
# NO_AVAILABLE_SPACE_OR_ON_AIR = -1000
# LESS_CROWDED_AREA_REWARD = 0
DWELL_VIOLATION_REWARD = -1.
# STACK_SORTING_DAMAGE = -20
# DWELL_COMPATIBLE_REWARD = 1

#run parameters
NUM_EPISODES = 10
TEST_EPISODES = 1


rsn = random.randint(10,99)

folder_path = f'ContainerAllocationRL/tf_encod_sortedMin/outputs/{BAYS}{ROWS}{TIERS}_{NUM_CONTAINERS_PER_EPISODE}_{rsn}'
os.makedirs(folder_path, exist_ok=True)

MODEL_PATH =f'{folder_path}/{rsn}_tf_encSrtMin_{BAYS}{ROWS}{TIERS}_{NUM_CONTAINERS_PER_EPISODE}_{datetime.now().strftime("%m_%d_%H_%M")}.mdl'
TRAIN_LOSS_REWARD_PATH =f'{folder_path}/loss_reward_{rsn}_{datetime.now().strftime("%m_%d_%H_%M")}.csv'
TEST_OPERATION_PATH =f'{folder_path}/test_{rsn}_{datetime.now().strftime("%m_%d_%H_%M")}.csv'






# MODEL_PATH =f'model_{rsn}_tf_encoding_isSortedMin_{BAYS}{ROWS}{TIERS}_{NUM_EPISODES}_{datetime.now().strftime("%Y_%m_%d__%H_%M")}.mdl'
# TRAIN_LOSS_REWARD_PATH =f'loss_reward_{rsn}_{datetime.now().strftime("%Y_%m_%d__%H_%M")}.csv'
# TEST_OPERATION_PATH =f'test_{rsn}_{datetime.now().strftime("%Y_%m_%d__%H_%M")}.csv'

DRAW_GRAPH = False

# from google.colab import auth
# auth.authenticate_user()

# np.random.seed(123)
# random.seed(123)

#dqn parameters
GAMMA = 0.98
LR = 0.001
BATCH_SIZE = 512
MEMORY_SIZE = 100_000
EPSILON_START = 1
EPSILON_END = 0.01
EPSILON_DECAY = 0.998
TARGET_UPDATE = NUM_CONTAINERS_PER_EPISODE * 20

class YardVisualizer:
    def __init__(self, bays, rows, tiers):
        self.bays = bays
        self.rows = rows
        self.tiers = tiers
    def set_yard(self, yard):
        self.yard = yard


    def draw_yard(self, allocated_cube_bay, allocated_cube_row, allocated_cube_tier, allocated_cube_color=(1, 1, 0, 1)):
        fig = plt.figure(figsize=(8, 4))
        self.ax = fig.add_subplot(111, projection='3d')

        # Iterate through the yard and draw cubes
        for z in range(self.tiers):
            for y in range(self.rows):
                for x in range(self.bays):
                    if self.yard[z, y, x] != 0:
                        # self.draw_cube(x, y, z)
                        self.draw_cube_2x(x=x, y=y, z=z)
                        self.add_text_label(x=x, y=y, z=z, dwell_time= self.yard[z, y, x])

        self.draw_cube_2x(x=allocated_cube_bay, y=allocated_cube_row, z=allocated_cube_tier,color=allocated_cube_color)
        # self.draw_cube(allocated_cube_bay, allocated_cube_row, allocated_cube_tier, allocated_cube_color)
        self.ax.set_xlabel('BAY')
        self.ax.set_ylabel('ROW')
        self.ax.set_zlabel('TIER')

        self.ax.set_xlim(0, 2 * self.bays + 1)
        self.ax.set_ylim(0, self.rows + 1)
        self.ax.set_zlim(0, self.tiers + 1)

        self.ax.set_box_aspect([4 * self.bays, 4 * self.rows, 2 * self.tiers])
        # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.tight_layout()
        plt.show()

    def draw_cube(self, x, y, z, color=(0, 0, 1, 1)):
        # Define the vertices of the cube
        vertices = [
            [x, y, z],
            [x + 1, y, z],
            [x + 1, y + 1, z],
            [x, y + 1, z],
            [x, y, z + 1],
            [x + 1, y, z + 1],
            [x + 1, y + 1, z + 1],
            [x, y + 1, z + 1]
        ]

        # Define the six faces of the cube
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[7], vertices[6], vertices[2], vertices[3]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]]
        ]

        # Create a 3D polygon collection
        poly3d = Poly3DCollection(faces, alpha=.25, linewidths=0.1, edgecolors='r')
        poly3d.set_facecolor(color)
        self.ax.add_collection3d(poly3d)

    def draw_cube_2x(self, x, y, z, color=(0, 0, 1, 1)):
        # Define the vertices of the cube with scaling
        vertices = [
            [2 * x, y, z],
            [2 * (x + 1), y, z],
            [2 * (x + 1), y + 1, z],
            [2 * x, y + 1, z],
            [2 * x, y, z + 1],
            [2 * (x + 1), y, z + 1],
            [2 * (x + 1), y + 1, z + 1],
            [2 * x, y + 1, z + 1]
        ]

        # Define the six faces of the cube
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[7], vertices[6], vertices[2], vertices[3]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]]
        ]

        # Create a 3D polygon collection
        poly3d = Poly3DCollection(faces, alpha=.20, linewidths=0.1, edgecolors='r')
        poly3d.set_facecolor(color)
        self.ax.add_collection3d(poly3d)

    def add_text_label(self, x, y, z, dwell_time):
        """ Adds text label showing dwell time at the center of the cube. """
        self.ax.text(2 * x + 1, y + 0.5, z + 0.5, str(dwell_time), color='black',
                     fontsize=8, ha='center', va='center')

def plot_q_values(qMaxValues, qMinValues):
    plt.figure(figsize=(10, 5))
    plt.plot(qMaxValues, label="Max Q-value", color="blue")
    plt.plot(qMinValues, label="Min Q-value", color="red")
    plt.xlabel("Training Steps")
    plt.ylabel("Q-value")
    plt.title("Max and Min Q-values Over Training")
    plt.legend()
    plt.show()
def plot_learning_progress(rewards, losses, epsilons):
    """ Plots reward, loss, and epsilon decay over episodes. """
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Plot Episode Rewards
    axs[0].plot(rewards, label="Total Reward per Episode", color='blue')
    axs[0].set_title("Rewards Over Time")
    axs[0].set_xlabel("Episodes")
    axs[0].set_ylabel("Total Reward")
    axs[0].legend()
    axs[0].grid(True)

    # Plot Loss Values
    axs[1].plot(losses, label="Loss per Episode", color='red')
    axs[1].set_title("Loss Over Time")
    axs[1].set_xlabel("Episodes")
    axs[1].set_ylabel("Loss Value")
    axs[1].legend()
    axs[1].grid(True)

    # Plot Epsilon Decay
    axs[2].plot(epsilons, label="Epsilon Decay", color='green')
    axs[2].set_title("Epsilon Decay Over Time")
    axs[2].set_xlabel("Episodes")
    axs[2].set_ylabel("Epsilon")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


class ContainerYardEnv:
    def __init__(self, bays=BAYS, rows=ROWS, tiers=TIERS):
        self.bays = bays
        self.rows = rows
        self.tiers = tiers
        self.queue_length = NUM_CONTAINERS_PER_EPISODE
        self.yard_state_length = (self.bays * self.rows * 3) + self.queue_length # 3 = top view, stack height, is sorted


    def action_to_bay_row(self,action):  # 2d action space, row and tier
        bay = action % BAYS
        row = action // BAYS
        return bay, row

    def reset(self, yard_container_count,cover_tier0=False):
        """ Resets the yard and initializes a random state for the number of containers. """
        self.yard_top_view = np.zeros((self.rows, self.bays), dtype=np.float32)
        self.stack_height = np.zeros((self.rows, self.bays), dtype=np.float32)
        self.is_stack_sorted = np.ones((self.rows, self.bays), dtype=np.float32)
        self.full_yard = np.full((self.rows, self.bays, self.tiers), fill_value=1., dtype=np.float32)

        self.random_yard_initialization(container_count=yard_container_count, cover_tier0=cover_tier0)
        self.container_queue = [self.generate_container() for _ in range(NUM_CONTAINERS_PER_EPISODE)]

        self.step_idx = 0
        return self.get_state(), self.get_valid_action_mask()

    def generate_container(self, is_empty=False):
        if is_empty:
            # allows for additional features beside dwell time
            return [-1.]
        return [np.random.randint(1, MAX_DWELL_DAYS + 1) / MAX_DWELL_DAYS] # Random dwell time between 0  and  1


    def get_state(self):
        # min stack dwell time + stack height + isSorted

        future_containers = np.stack(
            self.container_queue
        )

        yard = np.reshape(self.full_yard, (BAYS * ROWS, TIERS))
        yard[yard==-1.] = 1. # Why!

        state = np.min(np.reshape(self.full_yard, (BAYS * ROWS, TIERS)), axis=-1)

        state = np.stack([state,
                          self.stack_height.flatten() / TIERS,
                          self.is_stack_sorted.flatten()], axis=-1)


        return {
            "field": np.float32(state),
            "queue": np.float32(future_containers)
        }


    def random_yard_initialization(self, container_count,cover_tier0):
        """ the rule in initialization is no container can be on the fly!!"""
        if not cover_tier0:
            count = 0
            while count < container_count:
                r = np.random.randint(0, self.rows)
                b = np.random.randint(0, self.bays)
                new_container = np.random.randint(1, MAX_DWELL_DAYS) / MAX_DWELL_DAYS

                if self.stack_height[r, b] < self.tiers:
                    if self.stack_height[r, b] > 0:
                        if self.yard_top_view[r, b] <= new_container: # violation of dwell time
                            self.is_stack_sorted[r, b] = 0  # when it is not sorted, it is not sorted anymore
                    self.yard_top_view[r,b] = new_container

                    self.full_yard[r, b,  int(self.stack_height[r,b])] = new_container

                    self.stack_height[r,b] += 1
                    count += 1

        else: # this initialization is used only for testing the ability of dwell time criteria learning
            for r in range(self.rows):
                for b in range(self.bays):
                    self.yard_top_view[r,b] = np.random.randint(1, MAX_DWELL_DAYS) / MAX_DWELL_DAYS
                    self.full_yard[r, b,  int(self.stack_height[r,b])] = self.yard_top_view[r,b]
                    self.stack_height[r,b] = 1

    def step(self, action):
        """ returns next_state, reward, done  """

        bay, row = self.action_to_bay_row(action)
        next_container = self.container_queue.pop(0)  # Remove first container from queue


        # why? (in the get state I fill the rest of queue with 0)
        self.container_queue.append(self.generate_container(is_empty=True))

        # should never get there
        if self.stack_height[row,bay] == self.tiers: # tier is full
            return self.get_state(), -100., False, self.get_valid_action_mask(),0  # tier does not exist


        reward, num_violations = self.calculate_reward(bay, row, next_container)

        if self.yard_top_view[row, bay] <= next_container:
            self.is_stack_sorted[row, bay] = 0

        self.yard_top_view[row, bay] = next_container[0]  # Store dwell time , non 0 means the slot is occupied
        self.full_yard[row, bay, int(self.stack_height[row, bay])] = self.yard_top_view[row, bay]

        self.stack_height[row, bay] += 1

        self.step_idx += 1
        done = self.step_idx == len(self.container_queue)

        return  self.get_state(), reward, done, self.get_valid_action_mask(), num_violations


    def calculate_reward(self, bay, row, container_dwell):

        reward = 0
        num_violations = 0

        # large penalty: Placing in a full bay (no valid tier)
        if self.stack_height[row,bay] == self.tiers : # tier is full .should be filtered out in step function
            raise ValueError("The row, bay here is impossible should be filtered out")

        # medium penalty: Bad stacking which shorter stay container is below :(
        if self.stack_height[row, bay] > 0: # stack is not empty
            if self.yard_top_view[row, bay] <= container_dwell:
                reward += DWELL_VIOLATION_REWARD
                num_violations += 1

        return reward, num_violations


    def get_valid_action_mask(self):
        not_full = self.stack_height < self.tiers
        return not_full.flatten()

    def action_to_bay_row_tier_for_state(self, action, state):
        stack_height = state["field"].reshape((self.rows, self.bays, -1))[:, :, 1]
        # state["field"] = min, height, sorted
        # second layer  ([:, :, 1]) is stack height

        bay,row = self.action_to_bay_row(action)

        new_tier = stack_height[row,bay] * TIERS  # stack_height[row,bay] is a fraction between 0 and 1 so to have int
        return new_tier, row, bay

    def state_to_3d_yard(self,state):
        yard3d = np.zeros((self.tiers, self.rows, self.bays), dtype=np.float32)
        stacks_count = self.rows * self.bays
        reshaped_state = state["field"].reshape((self.rows, self.bays, -1))
        top_view = reshaped_state[:, :, 0]
        stack_height = reshaped_state[:, :, 1] * TIERS

        for r in range(self.rows):
            for b in range(self.bays):
                if stack_height[r,b] > 0:
                    yard3d[0: int(stack_height[r,b]) -1 ,r,b] = -1
                    yard3d[int(stack_height[r,b]) -1 ,r,b] = top_view[r,b]

        return yard3d

class DQN(snt.Module):
    def __init__(self, sample_obs):
        super().__init__(name="DQN")

        self.encoder = snt.nets.MLP([16, 16], name="encoder", activate_final=True)

        self.q_mlp = snt.nets.MLP([32, 32, 1], name="q_mlp")

        # init the model weights (eager mode)
        batched_sample = tree.map_structure(
            lambda obs: obs[np.newaxis],
            sample_obs
        )
        self(batched_sample)
        # tree.map_structure : This applies a function recursively to every leaf (non-container object) in a nested structure
        # add a batch dimension to a nested data structure
            # obs = np.array([1, 2, 3])  shape: [3]
            # obs[np.newaxis] -> [[1, 2, 3]]
        # ::me      with above :
        # sample_obs = {
        #     "field": [6, 3],
        #     "queue": [4, 1]
        # }
        #becomes
        #batched_sample = {
        #     "field": [1, 6, 3],
        #     "queue": [1, 4, 1]
        # }

    def __call__(self, x):

        # shape (batch_dims..., Row x Bay, num_features    ::me    [min, isSorted, stack size] ) =  (Batch, 4*3, 3)
        field = x["field"]
        # shape (batch_dims..., queue length, num_features)   ::me  ex. (Batch,20,1)
        queue = x["queue"]

        batch_size = tf.shape(field)[0]
        num_stacks = BAYS * ROWS

        # Encode each stack independently with the queue as context
        # ::me  or repeat queue context to each stack
        # :: so for instance ROWS = 3, BAYS = 2 -> num_stacks = 6 , queue length = 4 , batch size = B
        # ::me  tf.tile(input, multiples, name=None) : repeats input multiples times
        # ::me  if for example queue is [B, 4, 1], then  tile(queue, [1, 6, 1])  repeats queue for each stack
        # ::me  now shape becomes [B, 24, 1]
        # ::me  now expand dim: expand_dims(..., axis=-1) -> shape: [B, 6, 4, 1, 1]

        incoming = tf.expand_dims(tf.tile(queue, [1, num_stacks, 1]), axis=-1) # [B, num_stacks * queue length, 1 , 1]
        incoming = tf.reshape(incoming, [batch_size, num_stacks, -1]) # [B, num_stacks, queue length * 8]
        # :me why in the comment above it is written B, num_stacks, queue length * 8], I think it should be [B, num_stacks, queue length]
        # :me from debug incoming shape is:  [1,54,243]  (B, row*bay , queue size)

        # add the queue to each stack as a context
        stack_queue_encodings = tf.concat([field, incoming], axis=-1) # :: me (now contatination ): field [B,54,3], incoming [B,54,243] -> [B,54,246]
        stack_queue_encodings = self.encoder(stack_queue_encodings) # ::me output size of encoder mpl: (B,54,16)

        context = tf.reshape(stack_queue_encodings, [batch_size, 1, -1]) # ::me context: (1,1,864)
        context = tf.tile(context, [1, num_stacks, 1]) # ::me tiling the encoded data of all stacks, context size : (B,54,864)

        # add the full field as a context to each stack
        q_input = tf.concat([stack_queue_encodings, context], axis=-1) #:: again adding each encoded stack again to the all encoded field
        #::me  now the input for the second mpl (q_mlp) has shape (B,54,864+16) = (B,54,880)

        # compute the values for each stack
        q_values = self.q_mlp(q_input) #::me  q_values shape : (B,54,1)
        q_values = tf.squeeze(q_values) #::me  (B,54)

        return q_values


    def get_weights(self) -> Dict[str, np.ndarray]:
        return {v.name: v.numpy()
                for v in self.trainable_variables}

    def set_weights(self, weights: Dict):
        x = {v.name: v for v in self.trainable_variables}
        for name, w in weights.items():
            x[name].assign(w)

    def save(self, path):
        with open(path, "wb+") as f:
            pickle.dump(self.get_weights(), f)

    def load(self, path):
        with open(path, "rb") as f:
            self.set_weights(pickle.load(f))



class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = env.yard_state_length # ::me in case (Bay row tier = 9 * 6 * 3 + Q (0.9 * 9*6*5) )
        self.action_size = env.bays * env.rows
        self.epsilon = EPSILON_START
        self.memory = deque(maxlen=MEMORY_SIZE)

        #::me self.env.reset returns self.get_state(), self.get_valid_action_mask()
        sample_obs, _ = self.env.reset(yard_container_count=int(env.tiers * env.rows * env.bays * INITIAL_YARD_OCCUPIED_RATIO),
                          cover_tier0=FILL_TIER0_AS_INITIALIZATION)
        self.q_network = DQN(sample_obs)
        self.target_network = DQN(sample_obs)

        self.target_network.set_weights(self.q_network.get_weights())
        self.optimizer = snt.optimizers.Adam(LR)

        self.num_train_steps = 0

    def select_action(self, state, mask):
        """ Choose an action using epsilon-greedy strategy. """
        if random.random() < self.epsilon:  # Exploration
            indices = np.where(mask)[0]
            return random.choice(indices)

        state = tree.map_structure(
            lambda obs: obs[np.newaxis],
            state
        )

        return self._compute_action(state, np.float32(mask))

    @tf.function
    def _compute_action(self, state, mask):
        Qs = self.q_network(state)
        mask = (1. - mask) * 10_000.
        # should automatically broadcast
        masked_Qs = Qs - mask
        return tf.argmax(masked_Qs, axis=-1)

    def train(self):
        """ Train the DQN using experience replay. """
        if len(self.memory) < BATCH_SIZE:
            return None

        batch = random.sample(self.memory, BATCH_SIZE)
        states, action_masks, actions, rewards, next_states, next_masks, dones = zip(*batch)

        # little trick to rearrange the list of dict observations into dict of batches
        states = tree.map_structure(
            lambda *obs: np.stack(obs),
            *states
        )
        next_states = tree.map_structure(
            lambda *obs: np.stack(obs),
            *next_states
        )

        next_masks = np.float32(np.stack(next_masks))

        actions = np.int32(np.stack(actions))
        rewards = np.float32(np.stack(rewards))
        dones = np.float32(np.stack(dones))


        loss = self._train(
            states, actions, rewards, next_states, next_masks, dones
        )

        self.num_train_steps += 1
        if self.num_train_steps % TARGET_UPDATE == 0:
            self.target_network.set_weights(self.q_network.get_weights())

        return loss.numpy()

    def discount_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    @tf.function
    def _train(self, states, actions, rewards, next_states, next_masks, dones):

        with tf.GradientTape() as tape:
            with tf.device("gpu:0"):
                q_values = self.q_network(states)
                batch_indices = tf.range(tf.shape(actions)[0])
                indices = tf.stack([batch_indices, actions], axis=1)
                q_taken = tf.gather_nd(q_values, indices)

                next_masks = (1. - next_masks) * 10_000.
                next_q_values = tf.stop_gradient(self.target_network(next_states)) - next_masks
                next_max_q = tf.reduce_max(next_q_values, axis=-1)

                targets = rewards + (GAMMA * next_max_q * (1 - dones))
                loss = tf.reduce_mean(tf.math.square(targets - q_taken))

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply(grads, self.q_network.trainable_variables)

        return loss


def run_dqn():
    env = ContainerYardEnv()
    agent = DQNAgent(env)
    episode_rewards = []
    episode_losses = []
    epsilon_values = []


    ema_loss = None
    ema_kpi = None
    ema_c = 0.95

    t = time.time()

    # pbar = trange(NUM_EPISODES, desc="Training", unit="ep")
    # for episode in pbar:
    for episode in range(NUM_EPISODES):
        state, action_mask = env.reset(yard_container_count=int(env.tiers * env.rows * env.bays * INITIAL_YARD_OCCUPIED_RATIO),
                          cover_tier0=FILL_TIER0_AS_INITIALIZATION)
        total_reward = 0
        total_violations = 0
        episode_loss = 0
        loss_count = 0

        for _ in range(NUM_CONTAINERS_PER_EPISODE):
            action = agent.select_action(state, action_mask)
            next_state, reward, done, next_mask, num_violations = env.step(action)
            agent.memory.append((state, action_mask, action, reward, next_state, next_mask, done))
            state = next_state
            action_mask = next_mask

            loss = agent.train()

            if loss is not None:
                episode_loss += loss
                loss_count += 1
            total_reward += reward
            total_violations += num_violations

        avg_loss = episode_loss / max(loss_count, 1)
        episode_losses.append(round(avg_loss,4))

        if ema_loss == None:
            ema_loss = avg_loss
            ema_kpi = total_violations / NUM_CONTAINERS_PER_EPISODE
        else:
            ema_loss = ema_c * ema_loss + (1-ema_c) * avg_loss
            ema_kpi = ema_c * ema_kpi + (1-ema_c) * (total_violations / NUM_CONTAINERS_PER_EPISODE)

        # its KPI overtime, not "rewards" here
        episode_rewards.append(total_violations / NUM_CONTAINERS_PER_EPISODE)
        epsilon_values.append(agent.epsilon)

        agent.discount_epsilon()
        # pbar.set_postfix(loss=ema_loss, KPI=ema_kpi, eps=agent.epsilon)
        print(f"Episode {episode}, Loss Avg:{ema_loss:.4f},  Reward = {total_reward}, Epsilon = {agent.epsilon:.4f}")

    agent.q_network.save(MODEL_PATH)
    hours, remainder = divmod(round(time.time()-t), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Time taken: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"\nModel saved to {MODEL_PATH}!")
    print(f"\nMax reward calculated:{max(episode_rewards)}\n")

    df = pd.DataFrame({
        "Loss": episode_losses,
        "Reward": episode_rewards
    })
    df.to_csv(TRAIN_LOSS_REWARD_PATH, index=False)


    if DRAW_GRAPH:
        plot_learning_progress(episode_rewards, episode_losses, epsilon_values)

def run_test_agent(model_path):
    env = ContainerYardEnv()
    agent = DQNAgent(env)
    if os.path.exists(model_path):
        agent.q_network.load(model_path)
        print("\n Loaded trained model!")

    agent.epsilon = 0  # Only exploitation

    operation = {"round": [], "step": [], "currentContainer": [], "tier": [], "row": [], "bay": [], "reward": []}
    print(f"number of tests: {TEST_EPISODES},  number of steps in each test:{NUM_CONTAINERS_PER_EPISODE}")
    for test in range(TEST_EPISODES):

        state, mask = env.reset(yard_container_count=int(env.tiers * env.rows * env.bays * INITIAL_YARD_OCCUPIED_RATIO),
                          cover_tier0=False)
        for stp in range(NUM_CONTAINERS_PER_EPISODE):  # multiple container per episode
            action = agent.select_action(state, mask)

            next_state, reward, done, next_mask, num_violations = env.step(action)

            tier, row, bay = env.action_to_bay_row_tier_for_state(action, state)

            if DRAW_GRAPH:
                visualizer = YardVisualizer(BAYS, ROWS, TIERS)

                yard3d = env.state_to_3d_yard(state=state)
                visualizer.set_yard(yard=yard3d)
                visualizer.draw_yard(allocated_cube_bay=bay,
                                     allocated_cube_row=row,
                                     allocated_cube_tier=tier,
                                     allocated_cube_color=(0, 1, 0, 1))

            operation["round"].append(test)
            operation["step"].append(stp)
            operation["currentContainer"].append(env.yard_top_view[row, bay])
            operation["tier"].append(int(tier))
            operation["row"].append( row.numpy()  )
            operation["bay"].append( bay.numpy() )
            operation["reward"].append(reward)

            print(f"step: {stp},  Test {test}: Placed container at ({bay}, {row}), Reward: {reward} \n")
            state=next_state
    df = pd.DataFrame(operation)
    df.to_csv(TEST_OPERATION_PATH, index=False)

start_time=time.time()
print(f"code starting at:{datetime.now()}")

#-----------------

run_dqn()

#-----------------

print(f"total learning process time:{timedelta(seconds= time.time() - start_time)}")
# check model path file name
run_test_agent(model_path=MODEL_PATH)