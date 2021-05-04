from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import _pick_unused_card, _fill_community_card, gen_cards
import numpy as np
import keras.backend as backend
from keras.models import Sequential, load_model
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

ANTE = 0.25
SMALL_BLIND_AMOUNT = .5
PLAYERS = 9
STREET = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
CARDS = {'HA': 0, 'H2': 1, 'H3': 2, 'H4': 3, 'H5': 4, 'H6': 5, 'H7': 6, 'H8': 7, 'H9': 8, 'HT': 9, 'HJ': 10, 'HQ': 11, 'HK': 12,
		'DA': 13, 'D2': 14, 'D3': 15, 'D4': 16, 'D5': 17, 'D6': 18, 'D7': 19, 'D8': 20, 'D9': 21, 'DT': 22, 'DJ': 23, 'DQ': 24, 'DK': 25,
		'SA': 26, 'S2': 27, 'S3': 28, 'S4': 29, 'S5': 30, 'S6': 31, 'S7': 32, 'S8': 33, 'S9': 34, 'ST': 35, 'SJ': 36, 'SQ': 37, 'SK': 38,
		'CA': 39, 'C2': 40, 'C3': 41, 'C4': 42, 'C5': 43, 'C6': 44, 'C7': 45, 'C8': 46, 'C9': 47, 'CT': 48, 'CJ': 49, 'CQ': 50, 'CK': 51, 'NA': 100} #H D S C
ACTIONS = {'FOLD': 1, 'CALL': 2, 'RAISE': 3}

ALWAYS_CALL = False

opponent_obj_ids = []
dict_opponent_obj_ids = {}
first_reset = True
global reset

LOAD_MODEL = 'models/3x256_Bot9__792.50max___16.15avg_-111.00min_1620131055.model'

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 1_000  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '3x256_Bot1' # For model save
MIN_REWARD = 100  
MEMORY_FRACTION = 0.20

# Exploration settings
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Create models folder
if not os.path.isdir('models'):
	os.makedirs('models')

# Estimate the ratio of winning games given the current state of the game
def estimate_win_rate(nb_simulation, nb_player, hole_card, community_card=None):
	if not community_card: community_card = []

	# Make lists of Card objects out of the list of cards
	community_card = gen_cards(community_card)
	hole_card = gen_cards(hole_card)

	# Estimate the win count by doing a Monte Carlo simulation
	win_count = sum([montecarlo_simulation(nb_player, hole_card, community_card) for _ in range(nb_simulation)])
	return 1.0 * win_count / nb_simulation


def montecarlo_simulation(nb_player, hole_card, community_card):
	# Do a Monte Carlo simulation given the current state of the game by evaluating the hands
	community_card = _fill_community_card(community_card, used_card=hole_card + community_card)
	unused_cards = _pick_unused_card((nb_player - 1) * 2, hole_card + community_card)
	opponents_hole = [unused_cards[2 * i:2 * i + 2] for i in range(nb_player - 1)]
	opponents_score = [HandEvaluator.eval_hand(hole, community_card) for hole in opponents_hole]
	my_score = HandEvaluator.eval_hand(hole_card, community_card)
	return 1 if my_score >= max(opponents_score) else 0

class DQNAgent:
	def __init__(self):
		#Main model (trained every step)
		self.model = self.create_model()

		#Target model (.predict every step)
		self.target_model = self.create_model()
		self.target_model.set_weights(self.model.get_weights())

		self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

		self.tensorboard = ModifiedTensorBoard(log_dir = f"logs/{MODEL_NAME}-{int(time.time())}")

		self.target_update_counter = 0

	def create_model(self):
		if LOAD_MODEL is not None:
			print(f"Loading {LOAD_MODEL}")
			model = load_model(LOAD_MODEL)
			print(f"Model {LOAD_MODEL} loaded!")

		else:
			model = Sequential()

			model.add(Dense(units=256, input_shape=(1,)))
			model.add(Activation("relu"))

			model.add(Dense(256, activation='relu'))
			model.add(Dense(256, activation='relu'))
			model.add(Dense(256, activation='relu'))

			model.add(Dense(53, activation="linear"))
			model.compile(loss="mse", optimizer=Adam(lr=0.0001), metrics=['accuracy'])

			'''
			model = Sequential([
				Dense(units=256, input_shape=(1,), activation='relu'),
				Dense(units=256, activation='relu'),
				Dense(units=256, activation='relu'),
				Dense(units=7, activation='linear') #Output layer
			])
			'''
			'''
			Outputs = Fold, Call, Min Raise, 1/2 Pot Raise, 3/4 Pot Raise, Pot Raise, All In
			'''
		return model

	def update_replay_memory(self, transition):
		self.replay_memory.append(transition)

	def update_replay_memory_win(self, amount):
		current_state, action, reward, new_state, done = self.replay_memory.pop()
		reward = amount + 10
		done = True
		self.update_replay_memory((current_state, action, reward, new_state, done))

	def update_replay_memory_action(self, updated_state):
		current_state, action, reward, new_state, done = self.replay_memory.pop()
		new_state = updated_state
		self.update_replay_memory((current_state, action, reward, new_state, done))

	def get_qs(self, state):
		return self.model.predict(np.array(state)/10_000)[0]

	def train(self, terminal_state):
		if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
			return

		minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

		current_states = np.array([transition[0] for transition in minibatch])/10_000
		current_qs_list = self.model.predict(current_states)

		new_current_states = np.array([transition[3] for transition in minibatch])/10_000
		future_qs_list = self.target_model.predict(new_current_states)

		# X is inputs, y is outputs
		X = []
		y = []

		for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
			if not done:
				max_future_q = np.max(future_qs_list[index])
				new_q = reward + DISCOUNT * max_future_q
			else:
				new_q = reward

			current_qs = current_qs_list[index]
			current_qs[action] = new_q

			X.append(current_state)
			y.append(current_qs)

		self.model.fit(x=np.array(X)/10_000, y=np.array(y), batch_size = MINIBATCH_SIZE, epochs=10, verbose=0, shuffle=True, callbacks=[self.tensorboard] if terminal_state else None)

		# Update to determine when to update target_model
		if terminal_state:
			self.target_update_counter += 1

		if self.target_update_counter > UPDATE_TARGET_EVERY:
			self.target_model.set_weights(self.model.get_weights())
			self.target_update_counter = 0

class DQN1(BasePokerPlayer):
	def __init__(self):
		super().__init__()
		self.wins = 0
		self.losses = 0
		self.opponents = []
		self.opponent_obj_ids = []
		self.preflop_history = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		self.flop_history = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		self.turn_history = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		self.river_history = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		self.hands_played = 0
		self.call = True
		self.opponent_histories = []
		self.opponent_obj_ids = []
		self.need_opponents = True

		self.current_round_action_history = []
		self.action_history_backlog = []

		self.reset = True

		self.prev_round = 0
		self.round = 0

		if self in opponent_obj_ids:
			pass
		else: opponent_obj_ids.append(self)

		self.agent = DQNAgent()

		self.ep_rewards = []
		self.episode_reward = 0
		self.epsilon = 1

	def adjust_opponent_dict(self, player_id):
		if self not in opponent_obj_ids:
			opponent_obj_ids.append(self)


	def print_table_info(self, hole_card, round_state, player_id):
		bot_data_dict = {'max_players': PLAYERS, 'ante': ANTE, 'small_blind': SMALL_BLIND_AMOUNT}
		bot_data = [PLAYERS, ANTE, SMALL_BLIND_AMOUNT]

		participating_players = 0

		for player in range(len(round_state['seats'])):
			if round_state['seats'][player]['state'] == 'participating':
				participating_players += 1
		
		bot_data_dict['participating_players'] = participating_players
		bot_data.append(participating_players)

		seat_pos = 0
		for seat in round_state['seats']:
			if seat['uuid'] == player_id:
				stack = seat['stack']
			else: seat_pos += 1

		dealer_btn_to_left = round_state['dealer_btn'] + PLAYERS - seat_pos
		bot_data_dict['dealer_btn_to_left'] = dealer_btn_to_left
		bot_data.append(dealer_btn_to_left)

		bot_data_dict['pot'] = round_state['pot']['main']['amount']
		bot_data.append(round_state['pot']['main']['amount'])
		bot_data_dict['street'] = STREET[round_state['street']]
		bot_data.append(STREET[round_state['street']])

		table_cards = round_state['community_card']
		table_cards_dict_list = []
		for card in round_state['community_card']:
			bot_data.append(CARDS[card])
			table_cards_dict_list.append(CARDS[card])
		for extra in range(5 - len(round_state['community_card'])):
			bot_data.append(100)
			table_cards_dict_list.append(100)
		bot_data_dict['community_cards'] = table_cards_dict_list

		hole_cards_dict_list = []
		hole_cards_dict_list.append(CARDS[hole_card[0]])
		hole_cards_dict_list.append(CARDS[hole_card[1]])
		bot_data_dict['hole_cards'] = hole_cards_dict_list
		bot_data.append(CARDS[hole_card[0]])
		bot_data.append(CARDS[hole_card[1]])

		#Save and generate actions on this round
		action_history, dict_history = self.save_hand_histories(round_state, player_id)
		bot_data = bot_data + action_history
		bot_data_dict['action_history'] = dict_history

		#print(bot_data_dict)

		return(bot_data)


		#print(f"{self}, {valid_actions}")
		#print(hole_card)
		#print(round_state)

	def declare_action(self, valid_actions, hole_card, round_state, player_id):

		#print(round_state)

		self.uuid = player_id
		self.adjust_opponent_dict(player_id)
		table_information = self.print_table_info(hole_card, round_state, player_id)

		self.hands_played += 1

		penalty = 0

		# Create Chance to do a random action
		if np.random.random() > self.epsilon:
			bot_action = np.argmax(self.agent.get_qs(table_information)) # Do action based on argmax of the q values
		else:
			bot_action = np.random.randint(0, 53) # Do a random action

		bot_action_list = ['fold', 'call', 'min', '2x', '3x', '4x', '5x', '6x', '7x', '8x', '9x', '10x', '11x', '12x', '13x', '14x', 
							'15x', '16x', '17x', '18x', '19x', '20x', '25x', '30x', '35x', '40x', '45x', '50x', '60x', '70x', '80x', 
							'90x', '100x', '110x', '120x', '130x', '140x', '150x', '175x', '200x', '225x', '250x', '275x', '300x', 
							'350x', '400x', '400x', '500x', '600x', '750x', '900x', '1000x', 'max']

		bot_action_name = bot_action_list[bot_action]

		raise_amount_options = [item for item in valid_actions if item['action'] == 'raise'][0]['amount']
		if bot_action == 0:
			can_call = len([item for item in valid_actions if item['action'] == 'call']) > 0
			if can_call:
				# If so, compute the amount that needs to be called
				call_amount = [item for item in valid_actions if item['action'] == 'call'][0]['amount']
			else:
				call_amount = 0
			action = 'call' if can_call and call_amount == 0 else 'fold'
			amount = 0
			amount_name = bot_action_name
			penalty = -10 if action == 'call' else 0
		elif bot_action == 1:
			can_call = len([item for item in valid_actions if item['action'] == 'call']) > 0
			if can_call:
				# If so, compute the amount that needs to be called
				call_amount = [item for item in valid_actions if item['action'] == 'call'][0]['amount']
			else:
				call_amount = 0
			action = 'call' if can_call and call_amount == 0 else 'fold'
			amount = call_amount
			amount_name = bot_action_name
			penalty = -10 if action == 'fold' else 0
		elif bot_action > 1:
			action = 'raise'
			amount = raise_amount_options[bot_action_name]
			amount_name = bot_action_name
			if amount > raise_amount_options['max']:
				penalty = -10
				amount = raise_amount_options['max']
		if amount == -1 and amount_name != 'max':
			penalty = -10

		step_reward = 0 - amount + penalty
		self.episode_reward += step_reward

		self.agent.update_replay_memory((np.array(table_information), action, step_reward, [], False))

		print(f"{self.uuid} {action} by {amount_name} for {amount}")

		return action, amount

	def save_hand_histories(self, round_state, player_id):

		if len(self.opponents) != PLAYERS:
			self.opponents = []
			my_place = 0
			for action in round_state['action_histories']['preflop']:
				if action['action'] == 'ANTE':
					self.opponents.append(action['uuid'])
			for op in self.opponents:
				if op == player_id:
					my_place = self.opponents.index(op)
			self.opponents = self.set_to_front_list(self.opponents, my_place)
			

		stacks = {}
		participating = {}
		for opp in round_state['seats']:
			cur_uuid = opp['uuid']
			stacks[cur_uuid] = opp['stack']
			participating[cur_uuid] = opp['state']

		self.current_round_action_history = []

		# Save histories for all players #stored_history = {'uuid': opponent, 'preflop': preflop, 'flop': flop, 'turn': turn, 'river': river}
		for opp in self.opponents:
			stored_history = {'uuid': opp, 'stack' : stacks[opp], 'state': participating[opp], 'preflop': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'flop': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'turn': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 'river': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
			if stored_history['state'] != 'participating' and stored_history['state'] != 'allin':
				if len(self.current_round_action_history) < len(self.opponents):
					self.current_round_action_history.append(stored_history)
				else:
					for i in self.current_round_action_history:
						if i['uuid'] == opp:
							position = self.current_round_action_history.index(i)
							self.current_round_action_history[position] = stored_history
							break
				continue
			if round_state['street'] == 'preflop' or round_state['street'] == 'flop' or round_state['street'] == 'turn' or round_state['street'] == 'river':
				preflop_history = round_state['action_histories']['preflop']
				for action in preflop_history: # I am using all actions need to setup a filter to only pull the current players actions
					if action['uuid'] == opp:
						if action['action'] == 'CALL' or action['action'] == 'RAISE':
							stored_history['preflop'].append(ACTIONS[action['action']])
						if action['action'] == 'FOLD':
							stored_history['preflop'].append(1)
							stored_history['preflop'].append(0)
							stored_history['preflop'].append(0)
						elif action['action'] == 'CALL' or action['action'] == 'RAISE':
							stored_history['preflop'].append(action['amount'])
							stored_history['preflop'].append(action['paid'])
						while len(stored_history['preflop']) > 15:
							del stored_history['preflop'][0]
			if round_state['street'] == 'flop' or round_state['street'] == 'turn' or round_state['street'] == 'river':
				flop_history = round_state['action_histories']['flop']
				for action in flop_history:
					if action['uuid'] == opp:
						if action['action'] == 'CALL' or action['action'] == 'RAISE':
							stored_history['flop'].append(ACTIONS[action['action']])
						if action['action'] == 'FOLD':
							stored_history['flop'].append(1)
							stored_history['flop'].append(0)
							stored_history['flop'].append(0)
						elif action['action'] == 'CALL' or action['action'] == 'RAISE':
							stored_history['flop'].append(action['amount'])
							stored_history['flop'].append(action['paid'])
						while len(stored_history['flop']) > 15:
							del stored_history['flop'][0]
			if round_state['street'] == 'turn' or round_state['street'] == 'river':
				turn_history = round_state['action_histories']['turn']
				for action in turn_history:
					if action['uuid'] == opp:
						if action['action'] == 'CALL' or action['action'] == 'RAISE':
							stored_history['turn'].append(ACTIONS[action['action']])
						if action['action'] == 'FOLD':
							stored_history['turn'].append(1)
							stored_history['turn'].append(0)
							stored_history['turn'].append(0)
						elif action['action'] == 'CALL' or action['action'] == 'RAISE':
							stored_history['turn'].append(action['amount'])
							stored_history['turn'].append(action['paid'])
						while len(stored_history['turn']) > 15:
							del stored_history['turn'][0]
			if round_state['street'] == 'river':
				river_history = round_state['action_histories']['river']
				for action in river_history:
					if action['uuid'] == opp:
						if action['action'] == 'CALL' or action['action'] == 'RAISE':
							stored_history['river'].append(ACTIONS[action['action']])
						if action['action'] == 'FOLD':
							stored_history['river'].append(1)
							stored_history['river'].append(0)
							stored_history['river'].append(0)
						elif action['action'] == 'CALL' or action['action'] == 'RAISE':
							stored_history['river'].append(action['amount'])
							stored_history['river'].append(action['paid'])
						while len(stored_history['river']) > 15:
							del stored_history['river'][0]

			'''
			if len(self.action_history_backlog) < len(self.opponents):
				new_stat_sheet = {'uuid': opp, 'preflop': {'hands': 0, 'fold': 0, 'call': 0, 'raise': 0, '3bet': 0, 'call3bet': 0, 'fold3bet': 0, '4bet': 0}, 
								'flop': {'fold': 0, 'bet': 0, 'raise': 0, 'donk': 0}, 
								'turn': {'fold': 0, 'bet': 0, 'raise': 0, 'donk': 0}, 
								'river': {'fold': 0, 'bet': 0, 'raise': 0, 'donk': 0},
								'percentages': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}  #20 stats to keep track of per opponent player
				updated_stats = self.calculate_hud_stats(new_stat_sheet, stored_history)
				self.action_history_backlog.append(updated_stats)
			'''

			if len(self.current_round_action_history) < len(self.opponents):
				self.current_round_action_history.append(stored_history)
			else:
				for i in self.current_round_action_history:
					if i['uuid'] == opp:
						position = self.current_round_action_history.index(i)
						self.current_round_action_history[position] = stored_history
						break

		serialized_history, dict_history = self.serialize_history(self.current_round_action_history, player_id, round_state['dealer_btn'], round_state)

		return serialized_history, dict_history

	def serialize_history(self, history, player_id, dealer, round_state):
		for opp in history:
			if opp['uuid'] == player_id:
				position = history.index(opp)
				updated_history = self.set_to_front_list(history, position)
				break

		updated_dict_history = {}
		serialized_history = []
		seat = 0
		for opp in updated_history:
			dict_history = [{'seat': seat, 'stack': opp['stack'], 'preflop': opp['preflop'], 'flop': opp['flop'], 'turn': opp['turn'], 'river': opp['river']}]
			updated_dict_history.update({f'Player_{seat}': dict_history})
			serialized_history.append(seat)
			seat += 1
			serialized_history.append(opp['stack'])
			serialized_history = serialized_history + opp['preflop']
			serialized_history = serialized_history + opp['flop']
			serialized_history = serialized_history + opp['turn']
			serialized_history = serialized_history + opp['river']

		return serialized_history, updated_dict_history

	def set_to_front_list(self, the_list, index_position):
		return the_list[index_position:] + the_list[:index_position]

	def receive_game_start_message(self, game_info):
		self.num_players = game_info['player_num']

	def receive_round_start_message(self, round_count, hole_card, seats):
		self.hole_cards = hole_card

	def receive_street_start_message(self, street, round_state):
		pass

	def receive_game_update_message(self, action, round_state):
		if action['player_uuid'] == self.uuid:
			updated_state = self.print_table_info(self.hole_cards, round_state, self.uuid)
			self.agent.update_replay_memory_action(np.array(updated_state))

	def receive_round_result_message(self, winners, hand_info, round_state):
		is_winner = self.uuid in [item['uuid'] for item in winners]
		self.wins += int(is_winner)
		self.losses += int(not is_winner)
		for winner in winners:
			if winner['uuid'] == self.uuid:
				self.agent.update_replay_memory_win(winner['stack'])
				self.episode_reward += winner['stack']

		round_num = round_state['round_count']
		self.ep_rewards.append(self.episode_reward)
		# Append episode reward to a list and log stats (every given number of episodes)
		if not round_num % AGGREGATE_STATS_EVERY or round_num == 1:
			average_reward = sum(self.ep_rewards[-AGGREGATE_STATS_EVERY:])/len(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
			min_reward = min(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
			max_reward = max(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
			self.agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=self.epsilon)

			# Save model, but only when min reward is greater or equal a set value
			if min_reward >= MIN_REWARD:
				self.agent.model.save(f'models/{MODEL_NAME}_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min_{int(time.time())}.model')

		if not round_num % 5000 or round_num == 999:
			average_reward = sum(self.ep_rewards[-AGGREGATE_STATS_EVERY:])/len(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
			min_reward = min(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
			max_reward = max(self.ep_rewards[-AGGREGATE_STATS_EVERY:])
			self.agent.model.save(f'models/{MODEL_NAME}_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min_{int(time.time())}.model')

		# Decay epsilon
		if self.epsilon > MIN_EPSILON:
			self.epsilon *= EPSILON_DECAY
			self.epsilon = max(MIN_EPSILON, self.epsilon)

		self.episode_reward = 0


class ModifiedTensorBoard(TensorBoard):

	# Overriding init to set initial step and writer (we want one log file for all .fit() calls)
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.step = 1
		self.writer = tf.summary.create_file_writer(self.log_dir)
		self.writer.set_as_default()
		self.all_summary_ops = tf.compat.v1.summary.all_v2_summary_ops()

	# Overriding this method to stop creating default log writer
	def set_model(self, model):
		self._train_dir = self.log_dir + '\\train'

	# Overrided, saves logs with our step number
	# (otherwise every .fit() will start writing from 0th step)
	def on_epoch_end(self, epoch, logs=None):
		self.update_stats(**logs)

	# Overrided
	# We train for one batch only, no need to save anything at epoch end
	def on_batch_end(self, batch, logs=None):
		pass

	def on_train_begin(self, logs=None):
		pass

	# Overrided, so won't close writer
	def on_train_end(self, _):
		pass

	# added for performance?
	def on_train_batch_end(self, _, __):
		pass

	# Custom method for saving own metrics
	# Creates writer, writes custom metrics and closes writer
	def update_stats(self, **stats):
		#self._write_logs(stats, self.step)
		for name, value in stats.items():
			with self.writer.as_default():
				tf.summary.scalar(name, value, self.step)

def setup_ai():
	return DataBloggerBot()