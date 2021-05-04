from pypokerengine.api.game import start_poker, setup_config
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import _pick_unused_card, _fill_community_card, gen_cards
import numpy as np
import keras.backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
from dqn1 import DQN1, ANTE, SMALL_BLIND_AMOUNT, PLAYERS
from dqn2 import DQN2
from dqn3 import DQN3
from dqn4 import DQN4
from dqn5 import DQN5
from dqn6 import DQN6
from dqn7 import DQN7
from dqn8 import DQN8
from dqn9 import DQN9

if not os.path.isdir('models'):
    os.makedirs('models')

MAX_ROUNDS = 100000
INITIAL_STACK = 100

gpu = False

if gpu:
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	print(f"Num GPUs Available: {len(physical_devices)}")
	tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == '__main__':
    # The stack log contains the stacks of the Data Blogger bot after each game (the initial stack is 100)
    stack_log = []

    #pokerbot = DeepQBot()

    config = setup_config(max_round=MAX_ROUNDS, initial_stack=INITIAL_STACK, ante=ANTE, small_blind_amount=SMALL_BLIND_AMOUNT)

    config.register_player(name=f"dqn1", algorithm=DQN1())
    config.register_player(name=f"dqn2", algorithm=DQN2())
    config.register_player(name=f"dqn3", algorithm=DQN3())
    config.register_player(name=f"dqn4", algorithm=DQN4())
    config.register_player(name=f"dqn5", algorithm=DQN5())
    config.register_player(name=f"dqn6", algorithm=DQN6())
    config.register_player(name=f"dqn7", algorithm=DQN7())
    config.register_player(name=f"dqn8", algorithm=DQN8())
    config.register_player(name=f"dqn9", algorithm=DQN9())

    game_result = start_poker(config, verbose=0)

    #stack_log.append([player['stack'] for player in game_result['players'] if player['uuid'] == blogger_bot.uuid])
    #print('Avg. stack:', '%d' % (int(np.mean(stack_log))))