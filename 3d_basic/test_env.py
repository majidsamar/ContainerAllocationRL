import sys
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
from datetime import datetime,timedelta

MODEL_PATH =f'ContainerAllocationRL/3d_basic/outputs/model_basic_000_0_{datetime.now().strftime("%Y_%m_%d__%H_%M")}.mdl'
TRAIN_LOSS_REWARD_PATH =f'ContainerAllocationRL/3d_basic/outputs/loss_reward_{datetime.now().strftime("%Y_%m_%d__%H_%M")}.csv'
TEST_OPERATION_PATH =f'ContainerAllocationRL/3d_basic/outputs/test_{datetime.now().strftime("%Y_%m_%d__%H_%M")}.csv'

try:
    with open(MODEL_PATH, "w") as f:
        f.write("Environment test successful!\n")
    with open(TRAIN_LOSS_REWARD_PATH, "w") as f:
        f.write("Environment test successful!\n")
    with open(TEST_OPERATION_PATH, "w") as f:
        f.write("Environment test successful!\n")

except Exception as e:
    print(f" Failed to write to files : {e}")
    sys.exit(1)
print('OK')