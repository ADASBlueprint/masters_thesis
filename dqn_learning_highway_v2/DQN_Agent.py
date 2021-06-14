from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, InputLayer
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from collections import deque
import tensorflow as tf
import random
from HighEnv import HighwayEnv
import HighEnv
import numpy as np
import time


LEARNING_RATE = 0.001
DISCOUNT = 0.9
REPLAY_MEMORY_SIZE = 2_000  # How many last steps to keep for model 
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
# NORMALIZER = [HighEnv.speedlimit]+ [HighEnv.speedlimit, HighEnv.max_dist]*4 +[1, HighEnv.speedlimit]
MODEL_NAME = '3x1500'

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
        
    # def _write_logs(self, logs, index):
    #     with self.writer.as_default():
    #         for name, value in logs.items():
    #             tf.summary.scalar(name, value, step=index)
    #             self.step += 1
    #             self.writer.flush()

# Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.priorities = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(InputLayer(input_shape= (HighEnv.OUTPUTSIZE,), batch_size=MINIBATCH_SIZE))  # takes our 28x28 and makes it 1x784
        model.add(Dense(1500, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
        model.add(Dense(1500, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
        model.add(Dense(1500, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
        model.add(Dense(HighEnv.ACTION_SPACE_SIZE, activation=None))  # our output layer. 10 units for 10 classes. Softmax for probability distribution   activation=tf.nn.softmax

        model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        #save new experiences with a high priority so it has a high chance of getting sampled in next batch
        self.priorities.append(max(self.priorities, default=1))

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities)**priority_scale
        sample_probabilities = scaled_priorities/sum(scaled_priorities)
        return sample_probabilities
    
    def get_importance(self, probabilities):
        importance = 1/len(self.replay_memory) * 1/probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized
    
    def update_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i]=abs(e) + offset

    # Trains main network every step during episode
    def train(self, terminal_state, step, epsilon,priority_scale=0.7):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.replay_memory)), k=MINIBATCH_SIZE, weights=sample_probs)
        importance = self.get_importance(sample_probs[sample_indices])
        minibatch = np.array(self.replay_memory)[sample_indices]

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)
        model_future_qs_list = self.model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            # if not done:
            #     max_future_q = np.max(future_qs_list[index])
            #     new_q = reward + DISCOUNT * max_future_q
            # else:
            #     new_q = reward

            if not done:
                next_action = np.argmax(model_future_qs_list[index])
                max_future_q = future_qs_list[index][next_action]
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = np.copy(current_qs_list[index])
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)
        errors = np.add.reduce(np.square(y - current_qs_list),axis=(1))
        self.update_priorities(sample_indices, errors)
        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), epochs = 3, batch_size=MINIBATCH_SIZE, verbose=0,sample_weight=importance**(1-epsilon), shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        # self.model.fit(np.array(X)/NORMALIZER, np.array(y), epochs=3, batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        state=np.array(state)
        return self.model.predict(state.reshape(-1, *state.shape))[0]
