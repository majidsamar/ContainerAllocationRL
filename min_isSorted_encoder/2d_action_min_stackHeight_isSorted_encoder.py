# 2d_action_space_top_view_and_stack_height_isSorted
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from datetime import datetime, timedelta
from ContainerAllocationRL.helper.visualization import YardVisualizer, plot_q_values, plot_learning_progress
from collections import deque

from ContainerAllocationRL.helper.logger import TimeLogger

logger = TimeLogger()

print("Is CUDA available?", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# dqn parameters


# yard parameters
INITIAL_STACK_MIN_DWELLTIME = 0
INITIAL_YARD_OCCUPIED_RATIO = 0
MAX_DWELL_DAYS = 20
BAYS = 4  # X-axis
ROWS = 2  # Y-axis
TIERS = 3  # Stack height

FILL_TIER0_AS_INITIALIZATION = False  # how to initialize the yard block

# rewards
NO_AVAILABLE_SPACE = -1000
LESS_CROWDED_AREA_REWARD = 0
DWELL_VIOLATION_REWARD = -10
STACK_SORTING_DAMAGE = -20
DWELL_COMPATIBLE_REWARD = 0

# run parameters
NUM_CONTAINERS_PER_EPISODE = round(0.9 * BAYS * ROWS * TIERS )
NUM_EPISODES = 3000
TEST_EPISODES = 100



GAMMA = 0.98
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 1_000_000
EPSILON_START = 1
EPSILON_END = 0.005
EPSILON_DECAY = 0.997
TARGET_UPDATE = NUM_CONTAINERS_PER_EPISODE * 20



# "C:\Users\MajidSamar\Desktop\DATA_backup on 25-10-2024 before onedrive activation chaos\pv\unibe\thz\source code\rl test\RL_Project\RL_Tests\containerAllocation\current_allocation\
# 2d_TV_SH_IS_2d_423_10000_2025_04_06.mdl"

rsn = random.randint(1, 1000)

folder_path = f'ContainerAllocationRL/min_isSorted_encoder/outputs/MSE_{BAYS}{ROWS}{TIERS}_{NUM_CONTAINERS_PER_EPISODE}_{rsn}'
os.makedirs(folder_path, exist_ok=True)

MODEL_PATH = f'{folder_path}/{rsn}_{datetime.now().strftime("%m_%d_%H_%M")}.mdl'
TRAIN_LOSS_REWARD_PATH = f'{folder_path}/loss_reward_{rsn}_{datetime.now().strftime("%m_%d_%H_%M")}.csv'
TEST_OPERATION_PATH = f'{folder_path}/test_{rsn}_{datetime.now().strftime("%m_%d_%H_%M")}.csv'

DRAW_GRAPH = True


np.random.seed(123)
random.seed(123)

class ContainerYardEnv:
    def __init__(self, bays=BAYS, rows=ROWS, tiers=TIERS):
        self.bays = bays
        self.rows = rows
        self.tiers = tiers
        self.queue_length = NUM_CONTAINERS_PER_EPISODE
        self.yard_state_length = (self.bays * self.rows * 3) + self.queue_length  # 3 = min in each stack, stack height, is sorted

    def action_to_bay_row(self, action):  # 2d action space, row and tier
        bay = action % BAYS
        row = action // BAYS
        return bay, row

    def reset(self, yard_container_count, cover_tier0=False):
        """ Resets the yard and initializes a random state for the number of containers. """
        self.stack_min = np.full((self.rows, self.bays), INITIAL_STACK_MIN_DWELLTIME, dtype=np.float32)

        self.stack_height = np.zeros((self.rows, self.bays), dtype=np.float32)
        self.is_stack_sorted = np.ones((self.rows, self.bays), dtype=np.float32)
        self.container_queue = [self.generate_container() for _ in range(NUM_CONTAINERS_PER_EPISODE)]
        return self.get_state()

    def generate_container(self):
        # return round(random.uniform(0, 1), 2) # Random dwell time between 0  and  1 and
        return np.random.randint(1, MAX_DWELL_DAYS) / MAX_DWELL_DAYS  # note discrete distribution
        # Random dwell time between 0  and  1 and (1 in excluded)

    def get_state(self):
        future_containers = np.array(self.container_queue)
        future_containers = np.pad(
            future_containers,
            (0, NUM_CONTAINERS_PER_EPISODE - len(future_containers)),
            'constant',
            constant_values=-1  # 0 is  very close to 1/20 which differs alot so -1 is selected as initial value
        )

        state = np.concatenate([self.stack_min.flatten(),
                                self.stack_height.flatten(),
                                self.is_stack_sorted.flatten(),
                                future_containers])
        return state

    def step(self, action):
        """ returns next_state, reward, done  """

        # 25  26   27  28  29   | 2
        # 20  21   22  23  24   | 1  row   - tier 1
        # 15  16   17  18  19   | 0
        # --------------------------
        # 10  11  12  13  14    | 2
        # 5   6   7   8   9     | 1  row   - tier 0
        # 0   1   2   3   4     | 0
        # ------- bay ---------
        # 0   1   2   3   4

        bay, row = self.action_to_bay_row(action)
        next_container = self.container_queue.pop(0)  # Remove first container from queue

        if self.stack_height[row, bay] == self.tiers:  # tier is full
            # should not reach here
            raise ValueError("The row, bay here is impossible. They should be filtered out!!")
            #return self.get_state(), NO_AVAILABLE_SPACE , False  # tier does not exist

        reward = self.calculate_reward(bay, row, next_container)

        # ------------ updating sorted ------------------
        if next_container >= self.stack_min[row, bay] and self.stack_height[row, bay] > 0:
            self.is_stack_sorted[row, bay] = 0

        # ------------ updating min ------------------
        if self.stack_min[row, bay] == INITIAL_STACK_MIN_DWELLTIME:
            self.stack_min[row, bay] = next_container
        else:
            if next_container < self.stack_min[row, bay]:
                self.stack_min[row, bay] = next_container
        # ------------ updating height ------------------
        self.stack_height[row, bay] += 1

        done = len(self.container_queue) == 0  # End episode when all containers are placed
        return self.get_state(), reward, done

    def calculate_reward(self, bay, row, container_dwell):

        reward = 0

        if self.stack_height[row, bay] == self.tiers:  # tier is full .should be filtered out in step function
            raise ValueError("The row, bay here is impossible should be filtered out")

        if self.stack_height[row, bay] > 0:  # stack is not empty
            if self.stack_min[row, bay] <= container_dwell: # the = in <= is required because the state has not updated
                # and we if a container is == it means another container not the container itself.
                reward += DWELL_VIOLATION_REWARD
                if self.is_stack_sorted[row, bay] == 1:
                    # now it won't be sorted anymore, the update of is_stack_sorted will be done on step function
                    reward += STACK_SORTING_DAMAGE
            else:
                reward += DWELL_COMPATIBLE_REWARD # can be rempved is not necessary
        else:  # stack is empty
            reward += DWELL_COMPATIBLE_REWARD  # can be rempved is not necessary

        return reward

    def get_valid_actions(self):
        not_full = self.stack_height < self.tiers
        valid_indices = np.where(not_full)
        actions = valid_indices[0] * self.bays + valid_indices[1]  # valid_indices[0] : row, valid_indices[1] : col
        return actions.tolist()

    def get_valid_actions_for_state(self, state):
        # unflatten the state and extract only stack height for valid action calculation
        # stack_min = state[:self.yard_state_length].reshape((self.rows, self.bays))
        # container_queue = state[2 * self.yard_state_length:]
        stacks_count = self.rows * self.bays
        stack_height = state[stacks_count:2 * stacks_count].reshape((self.rows, self.bays)) #remark: min/ height/ sorted

        not_full = stack_height < self.tiers
        valid_indices = np.where(not_full)
        actions = valid_indices[0] * self.bays + valid_indices[1]
        return actions.tolist()

    def action_to_bay_row_tier_for_state(self, action, state):
        stacks_count = self.rows * self.bays
        stack_height = state[stacks_count:2 * stacks_count].reshape((self.rows, self.bays))

        bay, row = self.action_to_bay_row(action)

        new_tier = stack_height[row, bay] + 1
        return new_tier, row, bay


# ------------attention like nn structure----------------------
class YardEncoder(nn.Module):
    def __init__(self, queue_size, stack_features_size, hidden_dim, encoded_dim):
        super().__init__()
        self.input_dim = queue_size + stack_features_size

        # torch remark: the expected input shape is [*, input_dim] where * can be any number of leading batch dimensions
        # pytorch's nn.Linear automatically applies the linear layer to the last dimension of the tensor.
        # as a result we have applied the nn to each stack independently
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoded_dim),
            nn.ReLU()
        )

    def forward(self, stacks_features, queue):
        """
        stacks_features: [B, Bay*Row, #of features per stack ]   eg.:[B, 50, 3]
        queue: eg. [B, 80]
        Returns: [B, Bay*Row, encoded_dim] eg. [B, 50, 20]
        """
        B, S, _ = stacks_features.shape # B: batch size, S = Bays*Rows (number of stacks), _ # of features per stack =3
        # tile queue to shape [batch_size, num_stacks, queue_size]
        queue_tiled = queue.unsqueeze(1).repeat(1, S, 1)  # ex: [B, 50, 80]
        x = torch.cat([stacks_features, queue_tiled], dim=-1)  # [B, 50, 83]
        return self.encoder(x)  # returns -> [B, 50, 20]


class StackQValueHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # one Q-value per stack
        )

    def forward(self, fused_stack_vectors):
        """
        fused_stack_vectors: [B, 50, 1020]
        Returns: Q-values: [B, 50]  (input_dim= 1020, output_dim=1)
        """
        q_values = self.mlp(fused_stack_vectors)  # [B, 50, 1]
        return q_values.squeeze(-1)  # [B, 50]


class DQN(nn.Module):
    def __init__(self, queue_size, num_stacks, num_stack_features, encoded_dim):
        super().__init__()
        self.num_stacks = num_stacks
        self.stack_encoder = YardEncoder(queue_size=queue_size,
                                         stack_features_size=num_stack_features,
                                         hidden_dim=64,
                                         encoded_dim=encoded_dim)

        # q_head, reads gets one yard encoded(with Q) + current stack encoded (withQ) and gets th
        self.q_head = StackQValueHead(input_dim= num_stacks * encoded_dim + encoded_dim,
                                      hidden_dim=64 )
        #ex: input_dim= num_stacks*encoded_dim + encoded_dim : 50 * 20 + 20 = 1020

    # def state_to_stack_queue(self,state):
    #     # this one is not batched state parsing.
    #     state = state.view(-1) # because the state has shape [1, ROWS, BAYS,3 + queue]
    #     stack_features = state[0: ROWS * BAYS * 3].reshape((ROWS, BAYS,3))
    #     queue = state[ROWS * BAYS * 3:].reshape((NUM_CONTAINERS_PER_EPISODE))
    #     return stack_features,queue

    def state_to_stack_queue(self, state):
        # TODO:: future temporary this function is defined here I will create a helper class and seperate the business logic from here
        """
        state: [B, state_dim]
        Returns:
          stack_features: [B, 50, 3]
          queue: [B, 80]
        """
        B = state.shape[0]
        stack_feat_len = ROWS * BAYS * 3

        #stack_features = state[:, :stack_feat_len].reshape(B, ROWS * BAYS, 3) #error this one is row major , so fills first column first
        stack_features = state[:, :stack_feat_len].reshape(B, 3, ROWS * BAYS).transpose(1, 2)

        queue = state[:, stack_feat_len:]  #:: TODO check again this part   do I need reshaping!
        return stack_features, queue

    def forward(self, state):
        """
        stack_features: [B, 50, 3]
        queue: [B, 80]
        Returns: Q-values: [B, 50]
        """
        # step 0: convert state to stack_features and queue
        # I keep this structure for more readability and being similar to other agents.
        stack_features, queue = self.state_to_stack_queue(state)
        # here stack_features [B, 50, 3] and  queue: [B, 80]


        # Step 1: encode each stack
        stack_encoded = self.stack_encoder(stack_features, queue)  # [B, 50, 20]

        # Step 2: yard encoding by flattening all stacks encodings
        yard_encoded = stack_encoded.reshape(stack_encoded.size(0), -1)  # [B, 1000]

        # Step 3: tile yard encoding for each stack
        yard_tiled = yard_encoded.unsqueeze(1).repeat(1, self.num_stacks, 1)  # [B, 50, 1000]

        # Step 4: concatenate local stack vector with yard vector
        fused = torch.cat([stack_encoded, yard_tiled], dim=-1)  # [B, 50, 1020]

        # Step 5: predict Q-values per stack
        return self.q_head(fused)  # [B, 50]


# ----------------------------------

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = env.yard_state_length
        self.action_size = env.bays * env.rows
        self.epsilon = EPSILON_START
        self.memory = deque(maxlen=MEMORY_SIZE)

        # self.q_network = DQNConv(output_dim= self.action_size, num_containers=self.env.queue_length).to(device)
        # self.target_network = DQNConv(output_dim= self.action_size, num_containers=self.env.queue_length).to(device)
        # self.q_network = DQN(self.state_size, self.action_size).to(device)


        self.q_network = DQN(queue_size=NUM_CONTAINERS_PER_EPISODE,
                             num_stacks=self.action_size,
                             num_stack_features=3,
                             encoded_dim=40).to(device)  # ( queue_size=80, num_stacks=50, stack_features=3):

        self.target_network = DQN(queue_size=NUM_CONTAINERS_PER_EPISODE,
                                  num_stacks=self.action_size,
                                  num_stack_features=3,
                                  encoded_dim=40).to(device)

        # note : pytorch automatically traverse abd handles this through its nn.Module system.
        # for both encoder and head network the state dictionary will be cloned.
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)
        self.qMaxValuesInABatch = []
        self.qMinValuesInABatch = []

        self.qMaxValuesInOneSample = []
        self.qMinValuesInOneSample = []

    def select_action(self, state):
        """ Choose an action using epsilon-greedy strategy. """
        valid_actions = self.env.get_valid_actions()
        if random.random() < self.epsilon:  # Exploration
            return random.choice(valid_actions)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        mask = torch.full((self.action_size,), float('-inf'), device=q_values.device)
        mask[valid_actions] = q_values[0, valid_actions]

        best_action_idx = torch.argmax(mask).item()  # Select best valid action
        return best_action_idx

    def train(self):
        """ Train the DQN using experience replay. """
        if len(self.memory) < BATCH_SIZE:
            return None

        start_time = time.time()

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)  # (state, action, reward, next_state, done)

        # Convert NumPy arrays to Tensors more efficiently
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)

        actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        next_states_np = np.array(next_states)
        next_states = torch.tensor(next_states_np, dtype=torch.float32, device=device)

        t2 = time.time()
        logger.add_time_log(code_block_name="unpack from memory", elapsed_time=t2 - start_time)
        q_values = self.q_network(states)  #

        # selects the predicted Q-values only for the actions that were actually taken in the sampled experiences.
        q_values_selected = q_values.gather(1, actions).squeeze(1)

        # Log for visualization
        self.qMaxValuesInABatch.append(q_values.max().item())
        self.qMinValuesInABatch.append(q_values.min().item())

        t3 = time.time()
        logger.add_time_log(code_block_name="read q values", elapsed_time=t3 - t2)
        with torch.no_grad():  # mask invalid moves in the next state
            next_q_values = self.target_network(
                next_states)
            valid_next_q_values = []
            for idx, nx_state_np in enumerate(next_states_np):  # iteration over the batch
                valid_moves = self.env.get_valid_actions_for_state(nx_state_np)
                if valid_moves: #note for the next state we should also eliminate invalid moves
                    valid_move_tensor = torch.tensor(valid_moves, dtype=torch.long, device=device)
                    q_vals = next_q_values[idx][valid_move_tensor]
                    valid_next_q_values.append(q_vals.max().item())
                else:
                    valid_next_q_values.append(0.0)

        t4 = time.time()
        logger.add_time_log(code_block_name="calculate max next q value", elapsed_time=t4 - t3)

        valid_next_q_values = torch.tensor(valid_next_q_values, dtype=torch.float32, device=device)
        target_q_values = rewards + (GAMMA * valid_next_q_values * (1 - dones))
        loss = F.mse_loss(q_values_selected, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        t5 = time.time()
        logger.add_time_log(code_block_name="calc loss and backward propagation", elapsed_time=t5 - t4)
        # clip gradients to avoid instability because we can have exploding gradiant event
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        return loss.item()


def run_dqn():
    env = ContainerYardEnv()
    agent = DQNAgent(env)
    episode_rewards = []
    episode_losses = []
    epsilon_values = []
    for episode in range(NUM_EPISODES):
        # state shape will be [BAY*ROW + queue lenght ,]
        state = env.reset(yard_container_count=int(env.tiers * env.rows * env.bays * INITIAL_YARD_OCCUPIED_RATIO),
                          cover_tier0=FILL_TIER0_AS_INITIALIZATION)
        total_reward = 0
        episode_loss = 0
        loss_count = 0

        for _ in range(NUM_CONTAINERS_PER_EPISODE):
            action = agent.select_action(state=state)
            next_state, reward, done = env.step(action)  # tier should be removed from the output list of the step func
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            loss = agent.train()

            if loss is not None:
                episode_loss += loss
                loss_count += 1
            total_reward += reward

        avg_loss = episode_loss / max(loss_count, 1)
        episode_losses.append(avg_loss)

        episode_rewards.append(total_reward)
        epsilon_values.append(agent.epsilon)

        if episode % TARGET_UPDATE == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())

        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)
        print(f"Episode {episode}, Loss Avg:{avg_loss:.4f},  Reward = {total_reward}, Epsilon = {agent.epsilon:.4f}")

    torch.save(agent.q_network.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}!")
    print(f"\nMax reward calculated:{max(episode_rewards)}\n")

    df = pd.DataFrame({
        "Loss": episode_losses,
        "Reward": episode_rewards
    })
    df.to_csv(TRAIN_LOSS_REWARD_PATH, index=False)

    if DRAW_GRAPH:
        plot_q_values(qMaxValues=agent.qMaxValuesInABatch, qMinValues=agent.qMinValuesInABatch)
        plot_learning_progress(episode_rewards, episode_losses, epsilon_values)


def run_test_agent(model_path):
    env = ContainerYardEnv()
    agent = DQNAgent(env)
    if os.path.exists(model_path):
        agent.q_network.load_state_dict(torch.load(model_path, weights_only=True))
        agent.q_network.eval()
        print("\n Loaded trained model!")

    agent.epsilon = 0  # Only exploitation

    operation = {"round": [], "step": [], "currentContainer": [], "tier": [], "row": [], "bay": [], "reward": []}
    for test in range(TEST_EPISODES):

        state = env.reset(yard_container_count=int(env.tiers * env.rows * env.bays * INITIAL_YARD_OCCUPIED_RATIO),
                          cover_tier0=False)
        for stp in range(NUM_CONTAINERS_PER_EPISODE):  # multiple container per episode
            action = agent.select_action(state=state)
            current_container_in_Q = env.container_queue[0] # it will be poped in the env.step function

            next_state, reward, done = env.step(action)

            tier, row, bay = env.action_to_bay_row_tier_for_state(action, state)

            print(f" Test {test}: Placed container at ({bay}, {row}), Reward: {reward} \n")
            operation["round"].append(test)
            operation["step"].append(stp)
            operation["currentContainer"].append(current_container_in_Q)
            operation["tier"].append(tier)
            operation["row"].append(row)
            operation["bay"].append(bay)
            operation["reward"].append(reward)


            state = next_state
    df = pd.DataFrame(operation)
    df.to_csv(TEST_OPERATION_PATH, index=False)


start_time = time.time()
print(f"code starting at:{datetime.now()}")

# -----------------

run_dqn()
logger.print_log()

# -----------------

print(f"total learning process time:{timedelta(seconds=time.time() - start_time)}")
# check model path file name
run_test_agent(model_path=MODEL_PATH)
