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
EPISODES = int(sys.argv[1]) #N
T = 100
SAVE_EVERY = int(sys.argv[1])
RAMDOM = "false"

#filenames
cwd = os.getcwd()

#define collision update
def update_collision(env):
    global collisions
    if env.collision:
            collisions.append(1)
    else:
        collisions.append(0)

#define metrics update
def update_metrics(env, current_state, action, episode,  step):
    global avg_speed_list
    global overtaking_list
    global cum_collision_rates
    global d1,v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6, actions, timestep
    #collisions
    update_collision(env)
    cum_collision_rates.append(sum(collisions[1:])/len(collisions[1:]))

    d1.append(current_state["d1"])
    v1.append(current_state["v1"])
    d2.append(current_state["d2"])
    v2.append(current_state["v2"])
    d3.append(current_state["d3"])
    v3.append(current_state["v3"])
    d4.append(current_state["d4"])
    v4.append(current_state["v4"])
    d5.append(current_state["d5"])
    v5.append(current_state["v5"])
    d6.append(current_state["d6"])
    v6.append(current_state["v6"])
    L.append(current_state["L"])
    va.append(current_state["va"])
    vacc.append(current_state["vacc"])
    actions.append(action)
    timestep.append(step)

    # if env.collision:
    #     avg_speed_list.append(0)
    #     overtaking_list.append(0)
    # else:
    #     distance = env.auto_pos*8
    #     avg_spd = distance/step
    #     avg_speed_list.append(avg_spd)
    
    #     if avg_spd > env.maxspeed_slow:
    #         overtaking_list.append(1)
    #     else:
    #         overtaking_list.append(0)
    

#save metrics
def save_metrics():
    # metrics_file = "validation/{}-{}".format("metrics_log", int(time.time()))+".csv"
    # with open(metrics_file, 'w', newline ='') as f:
    #     writer = csv.writer(f)
    #     # writer.writerows(map(lambda x: [x], cum_collision_rates))
    #     writer.writerows(zip(collisions, cum_collision_rates, avg_speed_list, overtaking_list, timestep))
    metrics_file = "validation/{}-{}".format("data_log", int(time.time()))+".csv"
    with open(metrics_file, 'w', newline ='') as f:
        writer = csv.writer(f)
        # writer.writerows(map(lambda x: [x], cum_collision_rates))
        writer.writerows(zip(d1, v1, d2, v2, d3, v3, d4, v4,d5, v5, d6,v6, L, va, vacc, actions,timestep, collisions))

#initialize metrics
collisions = ["coll"]
cum_collision_rates = ["coll_rate"]
ep_rewards = ["ep_reward"]
overtaking_list = ["overtake"]
avg_speed_list = ["avg_spd"]
d1 = ["d1"]
v1 = ["v1"]
d2 = ["d2"]
v2 = ["v2"]
d3 = ["d3"]
v3 = ["v3"]
d4 = ["d4"]
v4 = ["v4"]
d5 = ["d5"]
v5 = ["v5"]
d6 = ["d6"]
v6 = ["v6"]
L = ["L"]
va = ["va"]
vacc = ["vacc"]
actions = ["actions"]
timestep = ["t"]

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

agent = create_model()   
# agent = tf.keras.models.load_model(os.path.join(cwd,r"models\3x1500__1608735473.model"))
agent.load_weights(os.path.join("models/agent.model"))

def main():
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
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
            # print(current_state)
            # print(action, qs)
            
            #step through environment w/ action
            new_state, reward, done, running = env.step(action)

            #update metrics
            if episode==int(sys.argv[1]):
                update_metrics(env, current_state, action, episode, step)
                
            if step == T:
                done = True
            elif not done:
                #update trackers and variables
                current_state = new_state.copy()
                step +=1
        # print(cum_collision_rates[-1])
        
        #save metrics
        if episode ==int(sys.argv[1]):
            save_metrics()

    env.close()

try:
    main()
except Exception as e:
    print(e)
    env.close()
