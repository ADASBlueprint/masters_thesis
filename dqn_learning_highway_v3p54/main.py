from tqdm import tqdm
import numpy as np
import HighEnv
from HighEnv import HighwayEnv
from DQN_Agent import DQNAgent
import DQN_Agent
import os
import time
import csv

#algo parameters
EPISODES = 7_000 #N
T = 100
EPSILON_DECAY = 0.9992
MIN_EPSILON = 0.1
AGGREGATE_STATS_EVERY = 2
SAVE_STATS_EVERY = 1_000
TRAIN_FREQ  = 5

#filenames
cwd = os.getcwd()


epsilon = 0.9

np.random.seed(1)

#define decay epsilon function
def update_epsilon():
     # Decay epsilon
    global epsilon
    epsilon *= EPSILON_DECAY
    epsilon = max(MIN_EPSILON, epsilon)

#define collision update
def update_collision(env):
    global collisions
    if env.collision:
            collisions.append(1)
    else:
        collisions.append(0)

#define metrics update
def update_metrics(env, episode, step, episode_reward, epsilon):
    global avg_speed_list
    global overtaking_list
    global ep_rewards
    #collisions
    update_collision(env)
    cum_collision_rates.append(sum(collisions)/episode)
    if not collisions[-1]:
        #overtaking
        distance = env.auto_pos*8
        avg_spd = distance/step
        avg_speed_list.append(avg_spd)
        if avg_spd > env.maxspeed_slow:
            overtaking_list.append(1)
        else:
            overtaking_list.append(0)
    else:
        if episode>1:
            avg_speed_list.append(avg_speed_list[-1])
            overtaking_list.append(overtaking_list[-1])
        else:
            avg_speed_list.append(0)
            overtaking_list.append(0)
    #rewards
    ep_rewards.append(episode_reward)
    agent.tensorboard.update_stats(collision_ratio=cum_collision_rates[-1], avg_spd=avg_speed_list[-1], overtake=overtaking_list[-1], reward=episode_reward, epsilon=epsilon)

#save metrics
def save_metrics():
    metrics_file = "metrics/{}-{}".format("metrics_log", int(time.time()))+".csv"
    with open(metrics_file, 'w', newline ='') as f:
        writer = csv.writer(f)
        # writer.writerows(map(lambda x: [x], cum_collision_rates))
        writer.writerows(zip(collisions, cum_collision_rates, avg_speed_list, ep_rewards))

#save model
def save_model():
    agent.model.save(f'models/{DQN_Agent.MODEL_NAME}__{int(time.time())}.model')

#update tensorboard
def update_tensorboard(episode, agent):
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        average_collision = sum(collisions[-AGGREGATE_STATS_EVERY:])/len(collisions[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon, collisions_avg = average_collision)

#initialize metrics
collisions = []
cum_collision_rates = []
ep_rewards = []
overtaking_list = []
avg_speed_list = []

#initialize environment and agent
env = HighwayEnv()
agent = DQNAgent()
counter = 0

def main ():
    global counter
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        # Reset environment, states, and trackers
        agent.tensorboard.step = episode
        current_state = env.reset()
        step = 1
        episode_reward = 0
        pseudo_idle = False
        running = False
        done = False
        while  not done:
            if not running:
                new_state, reward, done, running = env.step(0)
                current_state = new_state.copy()
                continue

            if np.random.random() > epsilon:
                qs = agent.get_qs(list(current_state.values()))
                action = np.argmax(qs)
            else:
                action = np.random.randint(0,env.ACTION_SPACE_SIZE)
            
            #step through environment w/ action
            if (current_state['L'] == 0 and action == 2) or (current_state['L'] == env.num_lanes-1 and action == 1):
                pseudo_idle = True
            else:
                pseudo_idle = False
            new_state, reward, done, running = env.step(action)

            if step == T:
                done = True

            counter+=1
			
            #store (st, at, rt, st+1, done) in memory buffer
            #store if collision avoidance is engaged
            if not((env.acc_counter >= 0 and env.decel_counter == 0) and new_state['vacc'] < 0) or (env.collision and step > 1):
                agent.update_replay_memory((list(current_state.values()), action, reward, list(new_state.values()), done))
            #train agent
            if counter==TRAIN_FREQ or env.collision:
                agent.train(done, step, epsilon)
                counter=0

            #update trackers and variables
            current_state = new_state.copy()
            step +=1
            episode_reward += reward

        #decay epsilon
        update_metrics(env, episode, step, episode_reward, epsilon)
        # update_tensorboard(episode, agent)
        update_epsilon()
        if not episode % SAVE_STATS_EVERY:
            save_metrics()
            save_model()
    env.close()

try:
    main()
except Exception as e:
    print(e)
    env.close()
