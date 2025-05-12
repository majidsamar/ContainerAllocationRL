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
from datetime import datetime,timedelta
from ContainerAllocationRL.helper.visualization import YardVisualizer,plot_q_values,plot_learning_progress
from collections import deque


from ContainerAllocationRL.helper.logger import TimeLogger
logger = TimeLogger()

print("Is CUDA available?", torch.cuda.is_available())
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
#dqn parameters
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 256
MEMORY_SIZE = 1_000_000
EPSILON_START = 1
EPSILON_END = 0.005
EPSILON_DECAY = 0.998
TARGET_UPDATE = 128

# yard parameters
INITIAL_YARD_OCCUPIED_RATIO = 0
MAX_DWELL_DAYS = 20
BAYS = 4  # X-axis
ROWS = 3  # Y-axis
TIERS = 3  # Stack height

FILL_TIER0_AS_INITIALIZATION = False # how to initialize the yard block

# rewards
NO_AVAILABLE_SPACE_OR_ON_AIR = -1000
# LESS_CROWDED_AREA_REWARD = 0

# base values
# DWELL_VIOLATION_REWARD = -10
# STACK_SORTING_DAMAGE = -20
# DWELL_COMPATIBLE_REWARD = 1
# STACK_TIGHT_REWARD_UNIT = 1


DWELL_VIOLATION_REWARD = -10
STACK_SORTING_DAMAGE = -20
DWELL_COMPATIBLE_REWARD = 1
STACK_TIGHT_REWARD_UNIT = 1

#run parameters
NUM_CONTAINERS_PER_EPISODE = 10
NUM_EPISODES = 200
TEST_EPISODES = 10
# "C:\Users\MajidSamar\Desktop\DATA_backup on 25-10-2024 before onedrive activation chaos\pv\unibe\thz\source code\rl test\RL_Project\RL_Tests\containerAllocation\current_allocation\
# 2d_TV_SH_IS_2d_423_10000_2025_04_06.mdl"

rsn = random.randint(1,1000)

folder_path = f'ContainerAllocationRL/topView_Sorted_HighRwd/outputs/{BAYS}{ROWS}{TIERS}_{NUM_CONTAINERS_PER_EPISODE}_{rsn}'
os.makedirs(folder_path, exist_ok=True)

MODEL_PATH =f'{folder_path}/{rsn}_TVSHR_{BAYS}{ROWS}{TIERS}_{NUM_CONTAINERS_PER_EPISODE}_{datetime.now().strftime("%m_%d_%H_%M")}.mdl'
TRAIN_LOSS_REWARD_PATH =f'{folder_path}/loss_reward_{rsn}_{datetime.now().strftime("%m_%d_%H_%M")}.csv'
TEST_OPERATION_PATH =f'{folder_path}/test_{rsn}_{datetime.now().strftime("%m_%d_%H_%M")}.csv'



DRAW_GRAPH = False


# from google.colab import auth
# auth.authenticate_user()

# np.random.seed(123)
# random.seed(123)

class ContainerYardEnv:
    def __init__(self, bays=BAYS, rows=ROWS, tiers=TIERS):
        self.bays = bays
        self.rows = rows
        self.tiers = tiers
        self.queue_length = NUM_CONTAINERS_PER_EPISODE
        self.yard_state_length = (self.bays * self.rows * 3) + self.queue_length # 3 = top view, stack height, is sorted

    @logger.log_time
    def action_to_bay_row(self,action):  # 2d action space, row and tier
        bay = action % BAYS
        row = action // BAYS
        return bay, row

    @logger.log_time
    def reset(self, yard_container_count,cover_tier0=False):
        """ Resets the yard and initializes a random state for the number of containers. """
        self.yard_top_view = np.zeros((self.rows, self.bays), dtype=np.float32)
        self.stack_height = np.zeros((self.rows, self.bays), dtype=np.float32)
        self.is_stack_sorted = np.ones((self.rows, self.bays), dtype=np.float32)

        self.random_yard_initialization(container_count=yard_container_count, cover_tier0=cover_tier0)
        self.container_queue = [self.generate_container() for _ in range(NUM_CONTAINERS_PER_EPISODE)]
        return self.get_state()

    @logger.log_time
    def generate_container(self):
        # return round(random.uniform(0, 1), 2) # Random dwell time between 0  and  1 and
        return np.random.randint(1, MAX_DWELL_DAYS) / MAX_DWELL_DAYS # Random dwell time between 0  and  1 and

    @logger.log_time
    def get_state(self):
        future_containers = np.array(self.container_queue)
        future_containers = np.pad(
            future_containers,
            (0, NUM_CONTAINERS_PER_EPISODE - len(future_containers)),
            'constant'
        )

        state = np.concatenate([self.yard_top_view.flatten(),
                                self.stack_height.flatten(),
                                self.is_stack_sorted.flatten(),
                                future_containers])
        return state

    @logger.log_time
    def random_yard_initialization(self, container_count,cover_tier0):

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
                    self.stack_height[r,b] += 1
                    count += 1
        else: # this initialization is used only for testing the ability of dwell time criteria learning
            for r in range(self.rows):
                for b in range(self.bays):
                    self.yard_top_view[r,b] = np.random.randint(1, MAX_DWELL_DAYS) / MAX_DWELL_DAYS
                    self.stack_height[r,b] = 1

    @logger.log_time
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

        if self.stack_height[row,bay] == self.tiers: # tier is full
            return self.get_state(), NO_AVAILABLE_SPACE_OR_ON_AIR, False  # tier does not exist

        reward = self.calculate_reward(bay, row, next_container)

        if self.yard_top_view[row, bay] <= next_container:
            self.is_stack_sorted[row, bay] = 0


        self.yard_top_view[row, bay] = next_container  # Store dwell time , non 0 means the slot is occupied
        self.stack_height[row, bay] += 1

        done = len(self.container_queue) == 0  # End episode when all containers are placed
        return self.get_state(), reward, done

    @logger.log_time
    def calculate_reward(self, bay, row, container_dwell):

        reward = 0

        # large penalty: Placing in a full bay (no valid tier)
        if self.stack_height[row,bay] == self.tiers : # tier is full .should be filtered out in step function
            raise ValueError("The row, bay here is impossible should be filtered out")

        # medium penalty: Bad stacking which shorter stay container is below :(
        if self.stack_height[row, bay] > 0: # stack is not empty
            if self.yard_top_view[row, bay] <= container_dwell:
                reward += DWELL_VIOLATION_REWARD
                if self.is_stack_sorted[row,bay] == 1:
                    # now it won't be sorted anymore, the update of is_stack_sorted will be done on step function
                    reward += STACK_SORTING_DAMAGE

            else:
                reward += self.stack_height[row, bay] * STACK_TIGHT_REWARD_UNIT
                reward += DWELL_COMPATIBLE_REWARD
        else: # stack is empty
            reward += DWELL_COMPATIBLE_REWARD

        return reward

    @logger.log_time
    def get_valid_actions(self):
        not_full = self.stack_height < self.tiers
        valid_indices = np.where(not_full)
        actions = valid_indices[0] * self.bays + valid_indices[1]
        return actions.tolist()

    @logger.log_time
    def get_valid_actions_for_state(self, state):
        # unflatten the state and extract only stack height for valid actoin calculation
        # yard_top_view = state[:self.yard_state_length].reshape((self.rows, self.bays))
        # container_queue = state[2 * self.yard_state_length:]
        stacks_count = self.rows * self.bays
        stack_height = state[stacks_count :2 * stacks_count].reshape((self.rows, self.bays))

        not_full = stack_height < self.tiers
        valid_indices = np.where(not_full)
        actions = valid_indices[0] * self.bays + valid_indices[1]
        return actions.tolist()

    @logger.log_time
    def action_to_bay_row_tier_for_state(self,action, state):
        stacks_count = self.rows * self.bays
        stack_height = state[stacks_count:2 * stacks_count].reshape((self.rows, self.bays))

        bay,row = self.action_to_bay_row(action)

        new_tier = stack_height[row,bay] + 1
        return new_tier, row, bay

    @logger.log_time
    def state_to_3d_yard(self,state):
        yard3d = np.zeros((self.tiers, self.rows, self.bays), dtype=np.float32)
        stacks_count = self.rows * self.bays
        top_view = state[0:stacks_count].reshape((self.rows, self.bays))
        stack_height = state[stacks_count:2 * stacks_count].reshape((self.rows, self.bays))

        for r in range(self.rows):
            for b in range(self.bays):
                if stack_height[r,b] > 0:
                    yard3d[0: int(stack_height[r,b]) -1 ,r,b] = -1
                    yard3d[int(stack_height[r,b]) -1 ,r,b] = top_view[r,b]

        return yard3d

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(128, 256)
        # self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(64, output_dim)

    @logger.log_time
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x  # Outputs Q-values for all actions

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = env.yard_state_length
        self.action_size = env.bays * env.rows
        self.epsilon = EPSILON_START
        self.memory = deque(maxlen=MEMORY_SIZE)

        # self.q_network = DQNConv(output_dim= self.action_size, num_containers=self.env.queue_length).to(device)
        # self.target_network = DQNConv(output_dim= self.action_size, num_containers=self.env.queue_length).to(device)
        self.q_network = DQN(self.state_size, self.action_size).to(device)
        self.target_network = DQN(self.state_size, self.action_size).to(device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)
        self.qMaxValuesInABatch=[]
        self.qMinValuesInABatch=[]

        self.qMaxValuesInOneSample = []
        self.qMinValuesInOneSample = []

    @logger.log_time
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

    @logger.log_time
    def train(self):
        """ Train the DQN using experience replay. """
        if len(self.memory) < BATCH_SIZE:
            return None

        start_time = time.time()

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)  #(state, action, reward, next_state, done)

        # Convert NumPy arrays to Tensors more efficiently
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)

        actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        next_states_np = np.array(next_states)
        next_states = torch.tensor(next_states_np, dtype=torch.float32, device=device)

        t2 = time.time()
        logger.add_time_log(code_block_name="unpack from memory",elapsed_time= t2-start_time)
        q_values = self.q_network(states)
        q_values_selected = q_values.gather(1, actions).squeeze(1)

        # Log for visualization
        self.qMaxValuesInABatch.append(q_values.max().item())
        self.qMinValuesInABatch.append(q_values.min().item())

        t3 = time.time()
        logger.add_time_log(code_block_name="read q values", elapsed_time=t3 - t2)
        with torch.no_grad():  # mask invalid moves in the next state
            next_q_values = self.target_network(next_states)
            valid_next_q_values = []
            for idx, nx_state_np in enumerate(next_states_np):# iteration over the batch
                valid_moves = self.env.get_valid_actions_for_state(nx_state_np)
                if valid_moves:
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
        #torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        return loss.item()

@logger.log_time
def run_dqn():
    env = ContainerYardEnv()
    agent = DQNAgent(env)
    episode_rewards = []
    episode_losses = []
    epsilon_values = []
    for episode in range(NUM_EPISODES):
        state = env.reset(yard_container_count=int(env.tiers * env.rows * env.bays * INITIAL_YARD_OCCUPIED_RATIO),
                          cover_tier0=FILL_TIER0_AS_INITIALIZATION)
        total_reward = 0
        episode_loss = 0
        loss_count = 0

        for _ in range(NUM_CONTAINERS_PER_EPISODE):
            action = agent.select_action(state= state)
            next_state, reward, done = env.step(action) #tier should be removed from the output list of the step func
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
        plot_q_values(qMaxValues=agent.qMaxValuesInABatch,qMinValues=agent.qMinValuesInABatch)
        plot_learning_progress(episode_rewards, episode_losses, epsilon_values)

def run_test_agent(model_path):
    env = ContainerYardEnv()
    agent = DQNAgent(env)
    if os.path.exists(model_path):
        agent.q_network.load_state_dict(torch.load(model_path,weights_only=True))
        agent.q_network.eval()
        print("\n Loaded trained model!")

    agent.epsilon = 0  # Only exploitation

    operation = {"round": [], "step": [], "currentContainer": [], "tier": [], "row": [], "bay": [], "reward": []}
    for test in range(TEST_EPISODES):

        state = env.reset(yard_container_count=int(env.tiers * env.rows * env.bays * INITIAL_YARD_OCCUPIED_RATIO),
                          cover_tier0=False)
        for stp in range(NUM_CONTAINERS_PER_EPISODE):  # multiple container per episode
            action = agent.select_action(state=state)
            next_state, reward, done = env.step(action)

            tier, row, bay = env.action_to_bay_row_tier_for_state(action,state)

            if DRAW_GRAPH:
                visualizer = YardVisualizer(BAYS, ROWS, TIERS)

                yard3d = env.state_to_3d_yard(state=state)
                visualizer.set_yard(yard=yard3d)
                visualizer.draw_yard(allocated_cube_bay=bay,
                                     allocated_cube_row=row,
                                     allocated_cube_tier=tier,
                                     allocated_cube_color=(0, 1, 0, 1))

            print(f" Test {test}: Placed container at ({bay}, {row}), Reward: {reward} \n")
            operation["round"].append(test)
            operation["step"].append(stp)
            operation["currentContainer"].append(env.yard_top_view[row, bay])
            operation["tier"].append(tier)
            operation["row"].append(row)
            operation["bay"].append(bay)
            operation["reward"].append(reward)

            print(f" Test {test + 1}: Placed container at ({bay}, {row}), Reward: {reward} \n")
            state=next_state
    df = pd.DataFrame(operation)
    df.to_csv(TEST_OPERATION_PATH, index=False)



start_time=time.time()
print(f"code starting at:{datetime.now()}")

#-----------------

run_dqn()
logger.print_log()

#-----------------

print(f"total learning process time:{timedelta(seconds= time.time() - start_time)}")
# check model path file name
run_test_agent(model_path=MODEL_PATH)
