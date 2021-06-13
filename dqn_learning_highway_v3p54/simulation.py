# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 17:00:38 2020

@author: JZST6G
"""

import HighEnv

env = HighEnv.HighwayEnv("false")

step = 1
reset = 1
pos = 0
var = 0
running = False
try: 
    # current_state = env.reset()
    while step <= 500:
        if running:
            action = int(input("Enter action: "))
            if (output['L'] == 0 and action == 2) or (output['L'] == env.num_lanes-1 and action == 1):
                action = 0
            output, reward, done, running = env.step(action)
            if not((action == 0 or action == 3) and output['vacc'] < 0) or (env.collision and step > 1):
                print("stored")
            print(output, env.acc)
            current_state = output.copy()
        else:
            output, reward, done, running = env.step(0)
        step+=1
    env.close()
except Exception as e:
    print(e)
    env.close()