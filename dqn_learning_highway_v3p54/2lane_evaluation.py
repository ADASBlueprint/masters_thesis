from tqdm import tqdm
import numpy as np
import HighEnv
from HighEnv import HighwayEnv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, InputLayer
from keras.optimizers import Adam
import tensorflow as tf
import os
import time
import csv
import sys



#algo parameters
EPISODES = 500 #int(sys.argv[1])
T = 100
SAVE_EVERY = EPISODES
RAMDOM = "false"
VERSION = sys.argv[1]

#filenames
cwd = os.getcwd()

#initialize environment and agent
env = HighwayEnv(RAMDOM)
# print(os.path.join(cwd,r"models\3x1500__1608735473.model"))

def create_model():
    model = Sequential()

    model.add(InputLayer(input_shape= (HighEnv.OUTPUTSIZE,)))  # takes our 28x28 and makes it 1x784
    model.add(Dense(1500, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
    model.add(Dense(1500, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
    model.add(Dense(1500, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
    model.add(Dense(HighEnv.ACTION_SPACE_SIZE, activation='linear'))  # our out
    return model

def update_spd(output):
    global cumulative_spd
    cumulative_spd.append(output['va'])

def update_acc(output):
    global cumulative_acc
    cumulative_acc.append(output['vacc'])

def update_avg_spd():
    global ep_avg_spd
    cumulative_spd_arr = np.array(cumulative_spd)
    ep_avg_spd.append(np.mean(cumulative_spd_arr))

def update_avg_acc():
    global ep_avg_acc
    cumulative_acc_arr = np.array(cumulative_acc)
    ep_avg_acc.append(np.mean(cumulative_acc_arr))

def update_std_dev():
    global ep_std_dev
    cumulative_spd_arr = np.array(cumulative_spd)
    ep_std_dev.append(np.std(cumulative_spd_arr))

def update_acc_std_dev():
    global ep_acc_std_dev
    cumulative_acc_arr = np.array(cumulative_acc)
    ep_acc_std_dev.append(np.std(cumulative_acc_arr))

def update_collison_rate(collisions):
    global cum_collision_rates
    cum_collision_rates.append(sum(collisions[1:])/len(collisions[1:]))

#define collision update
def update_collision(env):
    global collisions
    if env.collision:
            collisions.append(1)
    else:
        collisions.append(0)

#define define action counter
def update_actions_counter(action_counter):
    global idle_counter,llc_counter,rlc_counter,spdup_counter,spddwn_counter
    idle_counter.append(action_counter[0])
    llc_counter.append(action_counter[1])
    rlc_counter.append(action_counter[2])
    spdup_counter.append(action_counter[3])
    spddwn_counter.append(action_counter[4])

#save metrics
def save_metrics():
    metrics_file = "data/{}-{}".format("data", VERSION)+".csv"
    with open(metrics_file, 'w', newline ='') as f:
        writer = csv.writer(f)
        # writer.writerows(map(lambda x: [x], cum_collision_rates))
        writer.writerows(zip(cum_collision_rates,idle_counter,llc_counter,rlc_counter,spdup_counter,spddwn_counter,ep_avg_spd, ep_std_dev, ep_avg_acc,ep_acc_std_dev,collisions))
    

#initialize metrics
cumulative_spd = []
cumulative_acc = []
cum_collision_rates = ["coll_rate"]
idle_counter = ['idle']
llc_counter = ['llc']
rlc_counter = ['rlc']
spdup_counter = ['spdup']
spddwn_counter = ['spddwn']
ep_avg_spd = ['avg_spd']
ep_std_dev = ['spd_std_dev']
ep_avg_acc = ['avg_acc']
ep_acc_std_dev = ['acc_std_dev']
collisions = ["coll"]

agent = create_model()   
# agent = tf.keras.models.load_model(os.path.join(cwd,r"models\3x1500__1608735473.model"))
agent.load_weights(os.path.join("models/agent.model"))
def main():
    global cumulative_spd, cumulative_acc
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        #initialize metrics
        cumulative_spd = []
        cumulative_acc = []
        action_counter = [0,0,0,0,0]
        # Reset environment, states, and trackers
        current_state = env.reset()
        step = 1
        running = False
        done = False
        while  not done:
            if not running:
                new_state, reward, done, running = env.step(0)
                current_state = new_state.copy()
                continue
            state = list(current_state.values())
            state=np.array(state)
            qs = agent.predict(state.reshape(-1, *state.shape))[0]
            action = np.argmax(qs)
            if action == 0:
                action_counter[0]= action_counter[0]+1
            elif action == 1:
                action_counter[1]= action_counter[1]+1
            elif action == 2:
                action_counter[2]= action_counter[2]+1
            elif action == 3:
                action_counter[3]= action_counter[3]+1
            else:
                action_counter[4]= action_counter[4]+1
            #step through environment w/ action
            new_state, reward, done, running = env.step(action)

            update_spd(current_state)
            update_acc(current_state)
               
            if step == T:
                done = True
            elif not done:
                #update trackers and variables
                current_state = new_state.copy()
                step +=1
        update_avg_spd()
        update_std_dev()
        update_avg_acc()
        update_acc_std_dev()
        update_collision(env)
        update_collison_rate(collisions)
        update_actions_counter(action_counter)
    save_metrics()
try:
    main()
except Exception as e:
    print(e)
    env.close()
