import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from datetime import datetime,timedelta

from ContainerAllocationRL.helper.visualization import YardVisualizer,plot_q_values,plot_learning_progress
from ContainerAllocationRL.helper.logger import TimeLogger
logger = TimeLogger()

from collections import deque
import os

print("Is CUDA available?", torch.cuda.is_available())
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
#dqn parameters
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 1_000_000
EPSILON_START = 1.0
EPSILON_END = 0.005
EPSILON_DECAY = 0.997
TARGET_UPDATE = 100

# yard parameters
INITIAL_YARD_OCCUPIED_RATIO = 0
MAX_DWELL_DAYS = 20
BAYS = 10  # X-axis
ROWS = 4  # Y-axis
TIERS = 3  # Stack height

FILL_TIER0_AS_INITIALIZATION = False # how to initialize the yard block

# rewards
NO_AVAILABLE_SPACE_OR_ON_AIR = -1000
LESS_CROWDED_AREA_REWARD = 0
DWELL_VIOLATION_REWARD = -10
DWELL_COMPATIBLE_REWARD = 1



#run parameters
NUM_CONTAINERS_PER_EPISODE = 5
NUM_EPISODES = 300
TEST_EPISODES = 2


# MODEL_PATH =(f'convolutional/models/conv_{BAYS}{ROWS}{TIERS}_{NUM_EPISODES}_{datetime.now().strftime("%Y_%m_%d__%H")}.mdl')

MODEL_PATH =f'ContainerAllocationRL/convolutional/outputs/model_conv2d_{BAYS}{ROWS}{TIERS}_{NUM_EPISODES}_{datetime.now().strftime("%Y_%m_%d__%H_%M")}.mdl'
TRAIN_LOSS_REWARD_PATH =f'ContainerAllocationRL/convolutional/outputs/loss_reward_conv2d_{datetime.now().strftime("%Y_%m_%d__%H_%M")}.csv'
TEST_OPERATION_PATH =f'ContainerAllocationRL/convolutional/outputs/test_conv2d_{datetime.now().strftime("%Y_%m_%d__%H_%M")}.csv'

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
        self.capacity = bays * rows * tiers
        self.queue_length = NUM_CONTAINERS_PER_EPISODE


    @logger.log_time
    def action_to_bay_row(self,action):  # 2d action space, row and tier
        bay = action % BAYS
        row = action // BAYS
        return bay, row
    def reset(self, yard_container_count,cover_tier0=False):
        """ Resets the yard and initializes a random state for the number of containers. """
        self.yard = np.zeros((self.tiers, self.rows, self.bays), dtype=np.float32)
        self.random_yard_initialization(container_count=yard_container_count, cover_tier0=cover_tier0)
        self.container_queue = [self.generate_container() for _ in range(NUM_CONTAINERS_PER_EPISODE)]
        return self.get_state()

    def generate_container(self):
        # return round(random.uniform(0, 1), 2) # Random dwell time between 0  and  1 and
        return np.random.randint(1, MAX_DWELL_DAYS) / MAX_DWELL_DAYS # Random dwell time between 0  and  1 and

    def get_state_size(self):
        return (self.tiers * self.rows * self.bays) + self.queue_length
    def get_state(self):
        """Returns the yard state as a flattened 1D vector + ALL container dwell time"""
        # flat_normal_yard = self.yard.transpose(2, 1, 0).reshape(-1) #matching the action numbering scheme
        flat_normal_yard = self.yard.flatten() # tier, row, bay
        future_containers = np.array(self.container_queue)
        # note : yard and future container list are originally normalized
        # Pad with zeros if queue is shorter than N_CONTAINERS_PER_EPISODE
        future_containers = np.pad(
            future_containers,
            (0, NUM_CONTAINERS_PER_EPISODE - len(future_containers)),
            'constant'
        )
        return np.concatenate([flat_normal_yard, future_containers])

    @logger.log_time
    def random_yard_initialization(self, container_count,cover_tier0):
        """ the rule in initialization is no container can be on the fly!!"""
        if not cover_tier0:
            count = 0
            while count < container_count:
                z = np.random.randint(0, self.tiers)
                y = np.random.randint(0, self.rows)
                x = np.random.randint(0, self.bays)
                if z == 0 or self.yard[:z, y, x].all(): # we can not put a container on the fly
                    if not self.yard[z, y, x]:  # already is not selected
                        self.yard[z, y, x] = np.random.randint(1, MAX_DWELL_DAYS) / MAX_DWELL_DAYS
                        count += 1
        else: # this initialization is used only for testing the ability of dwell time criteria learning
            for i in range(self.bays):
                for j in range(self.rows):
                    self.yard[0, j, i] = np.random.randint(1, MAX_DWELL_DAYS) / MAX_DWELL_DAYS

    def step(self, action):
        """ returns next_state, reward, tier ,done  """

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
        next_container = self.container_queue.pop(0)

        stack = self.yard[:, row, bay]
        stack_height = (stack != 0).sum()

        if stack_height == self.tiers: # tier is full
            return self.get_state(), NO_AVAILABLE_SPACE_OR_ON_AIR, self.tiers, False

        self.yard[stack_height, row, bay] = next_container  # Store dwell time , non 0 means the slot is occupied
        reward = self.calculate_reward(bay, row, stack_height, next_container)# stack_height is the tier for putting
        done = len(self.container_queue) == 0  # End episode when all containers are placed
        return self.get_state(), reward, stack_height, done

    def calculate_reward(self, bay, row, tier, container_dwell):
        """ reward function for good placement. """

        reward = 0

        # large penalty: Placing in a full bay (no valid tier)
        if tier >= self.tiers: # should be filtered out in step function
            raise ValueError("The tier here is impossible should be filtered out")

        # medium penalty: Bad stacking which shorter stay container is below :(
        for t in range(tier):
            if self.yard[t, row, bay] <= container_dwell:
                reward += DWELL_VIOLATION_REWARD

        # small reward: Correct stacking which longer stay below shorter stay :)
        for t in range(tier):
            if self.yard[t, row, bay] > container_dwell:
                reward += DWELL_COMPATIBLE_REWARD
        if tier == 0:
            reward += DWELL_COMPATIBLE_REWARD
        # Bonus: Reward for placing in less crowded areas (encourages spreading)
        avg_stack_height = np.mean(np.sum(self.yard > 0, axis=0))
        if np.count_nonzero(self.yard[:, row, bay]) < avg_stack_height:
            reward += LESS_CROWDED_AREA_REWARD  # encourage placing in less filled areas

        return reward

    # def get_valid_actions(self):
    #     """ Returns  ( bay * row ) locations with available space.later it will convert into bay,row """
    #     valid_actions= []
    #     for t in range(TIERS):
    #         for r in range(ROWS):
    #             for b in range(BAYS):
    #                 if self.yard[t,r,b] == 0:
    #                     valid_actions.append(t*BAYS*ROWS +  r*BAYS + b)
    #                     break # to not calculate on the air cells
    #     return valid_actions
    @logger.log_time
    def get_valid_actions(self):
        # comments like below but here we do not need reshaping since the self.yard is already 3D
        occupied = self.yard > 0
        empty = ~occupied
        full_columns = ~np.any(empty, axis=0)
        valid_mask = ~full_columns
        row_indices,bay_indices  = np.where(valid_mask)
        actions = row_indices * BAYS + bay_indices

        return actions.tolist()



    @logger.log_time
    def get_valid_actions_for_state(self, state):
        yard_flat = state[:self.capacity]
        yard = yard_flat.reshape(self.tiers, self.rows, self.bays)

        occupied = yard > 0
        empty = ~occupied
        full_columns = ~np.any(empty, axis=0)
        valid_mask = ~full_columns
        row_indices, bay_indices = np.where(valid_mask)
        actions = row_indices * BAYS + bay_indices

        return actions.tolist()

class DQN(nn.Module):
    def __init__(self, output_dim, num_containers):
        super(DQN, self).__init__()

        self.conv1_out_channels = 8
        self.conv2_out_channels = 16

        self.conv1 = nn.Conv2d(TIERS, self.conv1_out_channels, kernel_size=(2, 2), stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.conv1_out_channels, self.conv2_out_channels, kernel_size=(2, 2), stride=1, padding=0)

        # output size for each layer will be reduced by 1 unit for each dimension
        # (since kernel_size - 1 for each spatial dimension without padding)

        conv_output_dims = (ROWS - 1, BAYS - 1)  # after first conv layer
        conv_output_dims = (conv_output_dims[0] - 1, conv_output_dims[1] - 1)  # after second conv layer

        self.feature_size = self.conv2_out_channels * conv_output_dims[0] * conv_output_dims[1] + num_containers


        # this 50 just for the output be larger than the input size
        self.fc1 = nn.Linear(self.feature_size, self.feature_size + 50)
        self.fc2 = nn.Linear(self.feature_size + 50, output_dim)

    @logger.log_time
    def forward(self, x):
        # Split the input into yard (reshaped for CNN) and queue components

        yard = x[:, :BAYS * ROWS * TIERS].view(-1, TIERS, ROWS, BAYS)
        queue = x[:, BAYS * ROWS * TIERS:]

        yard = F.relu(self.conv1(yard))
        yard = F.relu(self.conv2(yard))

        yard = yard.view(yard.size(0), -1)

        combined_features = torch.cat((yard, queue), dim=1)

        x = F.relu(self.fc1(combined_features))
        x = self.fc2(x)
        return x


class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = env.get_state_size()
        self.action_size = env.bays * env.rows
        self.epsilon = EPSILON_START
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.q_network = DQN(output_dim= self.action_size, num_containers=self.env.queue_length).to(device)
        self.target_network = DQN(output_dim= self.action_size, num_containers=self.env.queue_length).to(device)
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

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)  #(state, action, reward, next_state, done)

        # Convert NumPy arrays to Tensors more efficiently
        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)

        actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        next_states_np = np.array(next_states)
        next_states = torch.tensor(next_states_np, dtype=torch.float32, device=device)

        q_values = self.q_network(states)
        q_values_selected = q_values.gather(1, actions).squeeze(1)

        # Log for visualization
        self.qMaxValuesInABatch.append(q_values.max().item())
        self.qMinValuesInABatch.append(q_values.min().item())

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

        valid_next_q_values = torch.tensor(valid_next_q_values, dtype=torch.float32, device=device)
        target_q_values = rewards + (GAMMA * valid_next_q_values * (1 - dones))
        loss = F.mse_loss(q_values_selected, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
        episode_start_time = time.time()
        state = env.reset(yard_container_count=int(env.capacity * INITIAL_YARD_OCCUPIED_RATIO),
                          cover_tier0=FILL_TIER0_AS_INITIALIZATION)
        total_reward = 0
        episode_loss = 0
        loss_count = 0

        for _ in range(NUM_CONTAINERS_PER_EPISODE):
            action = agent.select_action(state= state)
            next_state, reward, _, done = env.step(action) #tier should be removed from the output list of the step func
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

        print(f"Episode {episode}, Loss Avg:{avg_loss:.4f},  Reward = {total_reward}, "
              f"Epsilon = {agent.epsilon:.4f} , Elapsed Time:{time.time() - episode_start_time:.6f} sec.")

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
        agent.q_network.load_state_dict(torch.load(model_path,weights_only=True))
        agent.q_network.eval()
        print("\n Loaded trained model!")

    agent.epsilon = 0  # Only exploitation
    operation = {"round": [], "step": [], "currentContainer": [], "tier": [], "row": [], "bay": [], "reward": []}

    for test in range(TEST_EPISODES):

        state = env.reset(yard_container_count=int(env.capacity * INITIAL_YARD_OCCUPIED_RATIO), cover_tier0=False)
        for stp in range(NUM_CONTAINERS_PER_EPISODE):  # multiple container per episode
            action = agent.select_action(state=state)
            next_state, reward, tier, done = env.step(action)
            bay, row = env.action_to_bay_row(action)

            if DRAW_GRAPH:
                visualizer = YardVisualizer(BAYS, ROWS, TIERS)
                visualizer.set_yard(env.yard)
                visualizer.draw_yard(allocated_cube_bay=bay,
                                     allocated_cube_row=row,
                                     allocated_cube_tier=tier,
                                     allocated_cube_color=(0, 1, 0, 1))

            print(f" Test {test}: Placed container at ({bay}, {row}), Reward: {reward}")

            operation["round"].append(test)
            operation["step"].append(stp)
            operation["currentContainer"].append(env.yard[tier, row, bay])
            operation["tier"].append(tier)
            operation["row"].append(row)
            operation["bay"].append(bay)
            operation["reward"].append(reward)

            state=next_state

    df = pd.DataFrame(operation)
    df.to_csv(TEST_OPERATION_PATH, index=False)

start_time=time.time()

print(f"code starting at:{datetime.now()}")
run_dqn()
logger.print_log()
print(f"total learing process time:{timedelta(seconds= time.time() - start_time)}")

# check model path file name
run_test_agent(model_path=MODEL_PATH)
