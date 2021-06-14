# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 12:44:35 2020

@author: Richard Hamilton
"""
import gym
import traci
import traci.constants as tc
import os, sys
from collections import OrderedDict 
from operator import getitem 

ENV = 2

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

if ENV == 0:
    sumoBinary = "/usr/bin/sumo"
    sumoconfig = "2lane_simulation.sumocfg"
elif ENV == 1:
    sumoBinary = os.path.normpath(r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe")
    # sumoconfig = os.path.normpath(r"W:\My Documents\McMaster\Thesis Project\model_analysis\2lane_simulation.sumocfg") 
    path = (os.path.dirname(os.path.abspath(__file__)))
    sumoconfig = path + r"\2lane_simulation.sumocfg"  
else:
    sumoBinary = os.path.normpath(r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe")
    path = (os.path.dirname(os.path.abspath(__file__)))
    sumoconfig = path + r"\2lane_simulation.sumocfg"   


#rewards
r_collision = -101
r_standstill = -50
r_near = -5
r_lanechange = 50
r_lane1 = -0.25
r_lane1approachvehicle = 0.5
r_overspeeding = -1
r_speeding = 1
r_maintainspeedlimit = 2

#parameters
d= 5  #seconds
speedlimit = 22.5 #m/s
NUM_LANES = 2
RANGE = 800
max_dist = 40
TARGET_ACC_DIFF_LIMIT = 0.5

def sort_nesteddict(dict,key):
    sort = OrderedDict(sorted(dict.items(), 
       key = lambda x: getitem(x[1], key)))
    return sort

#update dictionary functions
def update_dictionary(updatee, updater):
    for key in updater:
        if key in updatee:
            updatee[key] = updater[key]

ACTION_SPACE_SIZE = 5
output_keys = ['va', 'v1', 'd1',  'v2', 'd2', 'v3', 'd3', 'v4', 'd4', 'v5', 'd5', 'v6', 'd6', 'L', 'vacc']
output_values = [0, 0, max_dist, 0, max_dist, 0, max_dist,0, max_dist, 0, max_dist, 0, max_dist, 0, 0]   
OUTPUTSIZE = len(output_values)

class HighwayEnv(gym.core.Env):
    def __init__(self, random_flag="true"):
        self.seed = 1
        self.random_flag = random_flag
        sumoCmd = [sumoBinary, "-c", sumoconfig, "--seed", str(self.seed), "--random="+ self.random_flag]
        traci.isLibsumo()
        traci.start(sumoCmd)
        self.output = dict(zip(output_keys,output_values))
        self.output_prev = self.output.copy()
        self.num_vehs = 0
        self.auto_idx = 0
        self.auto_pos = 0
        self.auto_length = 0
        self.maxspeed = traci.vehicletype.getMaxSpeed("Auto")
        self.maxspeed_slow = traci.vehicletype.getMaxSpeed("SlowCar")
        self.rewards=0
        self.test = True
        self.sort_vehs = []
        self.target_spd=5
        # traci.vehicle.setSpeedMode("Auto",0x1F)
        # traci.vehicle.setLaneChangeMode("Auto",0x0)
        self.ACTION_SPACE_SIZE = 5
        self.SPEED_LIMIT = speedlimit
        self.MAX_DISTANCE = max_dist
        self.default = self.output
        self.acc = 1.26
        self.decel = - 0.63
        self.next_acc = 0
        self.acc_counter = 0
        self.decel_counter = 0
        self.num_lanes = NUM_LANES
        self.lane_change = False
        self.collision = False
        self.running = False
        self.done = False
#        traci.vehicle.setLaneChangeMode("Auto",0x0)
#        traci.vehicle.deactivateGapControl("Auto")
    def target_assignment(self, veh, v_key, d_key):
        self.output[v_key] = 0
        self.output[d_key] = max_dist
        d1 = veh[1][tc.VAR_POSITION][0] - self.auto_pos
        vt = self.output[v_key] = veh[1][tc.VAR_SPEED]
        va = self.output['va']
        spd_diff = va - vt
        if d1 >= 0:
            if d1 < RANGE and va > 0:
                ttc = d1/va
                if ttc < max_dist:
                    self.output[v_key] = vt
                    self.output[d_key] = ttc
        else:
            if d1 < RANGE and vt > 0:
                ttc = abs(d1/vt)
                if ttc < max_dist:
                    self.output[v_key] = vt
                    self.output[d_key] = ttc
                
            
    def find_forward_vehs(self, sort_vehicles):
        d1_found = 0
        d3_found = 0
        d5_found = 0
        idx = self.auto_idx+1

        #if Auto is in lane 0
        if self.output['L'] == 0:
            #update v5 and d5 to default values (no vehicle to the right)
            self.output['v5'] = 0
            self.output['d5'] = max_dist
            d5_found = 1
            #iterate through sorted vehicle lists for forward vehicles
            while idx < self.num_vehs and (not d1_found or not d3_found):
                veh = list(sort_vehicles.items())[idx]
                veh_lane = veh[1][tc.VAR_LANE_ID]
                #update in lane forward vehicle info
                if not d1_found and veh_lane == "Lane_0":
                    self.target_assignment(veh, 'v1', 'd1')
                    d1_found = 1
                #update left lane forward vehicle info
                elif not d3_found and veh_lane == "Lane_1":
                    self.target_assignment(veh, 'v3', 'd3')
                    d3_found = 1
                idx+=1
            if not d1_found:
                self.output['v1'] = 0
                self.output['d1'] = max_dist
            if not d3_found:
                self.output['v3'] = 0
                self.output['d3'] = max_dist
        #if Auto is in lane 1
        elif self.output['L'] == 1:
            #iterate through sorted vehicle lists for forward vehicles
            while idx < self.num_vehs and (not d1_found or not d3_found or not d5_found):
                veh = list(sort_vehicles.items())[idx]
                veh_lane = veh[1][tc.VAR_LANE_ID]
                #update in lane forward vehicle info
                if not d1_found and veh_lane == "Lane_1":
                    self.target_assignment(veh, 'v1', 'd1')
                    d1_found = 1
                #update left lane forward vehicle info
                elif not d3_found and veh_lane == "Lane_2":
                    self.target_assignment(veh, 'v3', 'd3')
                    d3_found = 1
                #update right lane forward vehcile info
                elif not d5_found and veh_lane == "Lane_0":
                    self.target_assignment(veh, 'v5', 'd5')
                    d5_found = 1
                idx+=1
            if not d1_found:
                self.output['v1'] = 0
                self.output['d1'] = max_dist
            if not d3_found:
                self.output['v3'] = 0
                self.output['d3'] = max_dist 
            if not d5_found:
                self.output['v5'] = 0
                self.output['d5'] = max_dist
        #if Auto is in lane 2        
        elif self.output['L'] == 2:
            #update v3 and d3 to default values (no vehicle to the left)
            self.output['v3'] = 0
            self.output['d3'] = max_dist
            d3_found = 1
            #iterate through sorted vehicle lists for forward vehicles
            while idx < self.num_vehs and (not d1_found or not d5_found):
                veh = list(sort_vehicles.items())[idx]
                veh_lane = veh[1][tc.VAR_LANE_ID]
                #update in lane forward vehicle info
                if not d1_found and veh_lane == "Lane_2":
                    self.target_assignment(veh, 'v1', 'd1')
                    d1_found = 1
                #update right lane forward vehcile info
                elif not d5_found and veh_lane == "Lane_1":
                    self.target_assignment(veh, 'v5', 'd5')
                    d5_found = 1
                idx+=1
            if not d1_found:
                self.output['v1'] = 0
                self.output['d1'] = max_dist
            if not d5_found:
                self.output['v5'] = 0
                self.output['d5'] = max_dist
    
    def find_trailing_vehs(self, sort_vehicles):
        d2_found = 0
        d4_found = 0
        d6_found = 0
        idx = self.auto_idx - 1

        #if Auto is in lane 0
        if self.output['L'] == 0:
            #update v6 and d6 to default values (no vehicle to the right)
            self.output['v6'] = 0
            self.output['d6'] = max_dist
            d6_found = 1
            #iterate through sorted vehicle lists for forward vehicles
            while idx >= 0 and (not d2_found or not d4_found):
                veh = list(sort_vehicles.items())[idx]
                veh_lane = veh[1][tc.VAR_LANE_ID]
                #update in lane following vehicle info
                if not d2_found and veh_lane == "Lane_0":
                    self.target_assignment(veh, 'v2', 'd2')
                    d2_found = 1
                #update left lane following vehicle info
                elif not d4_found and veh_lane == "Lane_1":
                    self.target_assignment(veh, 'v4', 'd4')
                    d4_found = 1
                idx-=1
            if not d2_found:
                self.output['v2'] = 0
                self.output['d2'] = max_dist
            if not d4_found:
                self.output['v4'] = 0
                self.output['d4'] = max_dist
        #if Auto is in lane 1
        elif self.output['L'] == 1:
            #iterate through sorted vehicle lists for following vehicles
            while idx >= 0 and (not d2_found or not d4_found or not d6_found):
                veh = list(sort_vehicles.items())[idx]
                veh_lane = veh[1][tc.VAR_LANE_ID]
                #update in lane following vehicle info
                if not d2_found and veh_lane == "Lane_1":
                    self.target_assignment(veh, 'v2', 'd2')
                    d2_found = 1
                #update left lane following vehicle info
                elif not d4_found and veh_lane == "Lane_2":
                    self.target_assignment(veh, 'v4', 'd4')
                    d4_found = 1
                #update right lane following vehicle info
                elif not d6_found and veh_lane == "Lane_0":
                    self.target_assignment(veh, 'v6', 'd6')
                    d6_found = 1
                idx-=1
            if not d2_found:
                self.output['v2'] = 0
                self.output['d2'] = max_dist
            if not d4_found:
                self.output['v4'] = 0
                self.output['d4'] = max_dist 
            if not d6_found:
                self.output['v6'] = 0
                self.output['d6'] = max_dist
        #if Auto is in lane 2
        elif self.output['L'] == 2:
            #update v4 and d4 to default values (no vehicle to the left)
            self.output['v4'] = 0
            self.output['d4'] = max_dist
            d4_found = 1
            #iterate through sorted vehicle lists for following vehicles
            while idx >= 0 and (not d2_found or not d6_found):
                veh = list(sort_vehicles.items())[idx]
                veh_lane = veh[1][tc.VAR_LANE_ID]
                #update in lane following vehicle info
                if not d2_found and veh_lane == "Lane_2":
                    self.target_assignment(veh, 'v2', 'd2')
                    d2_found = 1
                #update right lane following vehicle info
                elif not d6_found and veh_lane == "Lane_1":
                    self.target_assignment(veh, 'v6', 'd6')
                    d6_found = 1
                idx-=1
            if not d2_found:
                self.output['v2'] = 0
                self.output['d2'] = max_dist
            if not d4_found:
                self.output['v4'] = 0
                self.output['d4'] = max_dist 
            if not d6_found:
                self.output['v6'] = 0
                self.output['d6'] = max_dist

    #execute idle
    def idle(self):
        if self.output['vacc'] >= 0:
            if self.output['va'] + self.output['vacc'] >= self.target_spd:
                self.target_spd = max(self.output['va'],self.target_spd)
                traci.vehicle.slowDown("Auto", self.target_spd, 1)
        else:
            if self.output['va'] + self.output['vacc'] <= self.target_spd:
                traci.vehicle.slowDown("Auto", self.target_spd, 1)

    #execute speed up
    def speed_up(self):
        acc_rate = (self.acc * self.acc_counter)
        spd = self.output['va'] + acc_rate
        if spd > self.maxspeed:
            spd = self.maxspeed
        self.target_spd = spd
        t = (self.maxspeed - self.output['va'])/acc_rate
        self.next_acc = (self.target_spd - self.output['va'])/t
        traci.vehicle.slowDown("Auto", self.target_spd, t)
    
    #execute speed down
    def speed_down(self):
        decel_rate =(self.decel * self.decel_counter)
        spd = self.output['va'] + decel_rate
        t = self.output['va']/abs(decel_rate)
        if spd < 0:
            spd = 0
            t = 1
        self.target_spd = spd
        self.next_acc = (self.target_spd - self.output['va'])/t
        traci.vehicle.slowDown("Auto", self.target_spd, t)
    
    #execute left lane change
    def change_left(self):
        if self.output['L'] < NUM_LANES-1:
            traci.vehicle.changeLane("Auto", self.output['L']+1, 1)
            self.lane_change = True
            self.idle()
        else:
            self.idle()        
            
    
    #execute right lane change
    def change_right(self):
        if self.output['L'] != 0:
            traci.vehicle.changeLane("Auto",self.output['L']-1, 1)
            self.lane_change = True
            self.idle() 
              
    #determine reward
    def reward(self):
        reward = 0
        #detect collision
        collided_vehicles = traci.simulation.getCollidingVehiclesIDList()
        if "Auto" in collided_vehicles:
            reward += r_collision
            self.collision = True
            self.done = True
            print(collided_vehicles)        
        #detect standstill
        elif round(self.output['va'],2) == 0:
            reward += r_standstill
        #detect if lane change is required
        elif self.output['L'] != NUM_LANES-1 and self.output['d1'] < d:
                reward += r_near
        elif self.output['L'] == 0 and self.output['d1'] > d:
            reward += 5
#        #positive reward for being in the overtaking lane when a vhicle in front at close distance
        elif self.output['L'] != 0 and self.output['d5'] < d and self.output['va'] > self.output_prev['va']:
            reward += r_lanechange - (self.output['d5']*4)
# #        #negative reward for staying in the overtaking lane unnecessarily
#         elif self.output['L'] != 0 and self.output['d5'] > d and self.output['d6'] > 20:
#             reward += r_lane1 * self.output['d5']/8
#        #positive reward if host is in overtaking lane and slows down to avoid colliding with vehicle in the same lane
        elif self.output['L'] == NUM_LANES-1 and self.output['d1'] < d and self.output['va'] < self.output_prev['va']:
            reward += r_lane1approachvehicle
#        #negative reward if host is in overtaking lane and speeds up when there is a target vehicle infront
        elif self.output['L'] == NUM_LANES-1 and self.output['d1'] < d and self.output['va'] > self.output_prev['va']:
            reward += -r_lane1approachvehicle
#        #negative reward for going over speed limit
        elif self.output['va'] > speedlimit:
            reward += r_overspeeding
#        #positive reward for host to accelerate until it reaches the speed limit 
        elif self.output['va'] > self.output_prev['va']:
            reward += r_speeding
#        #positive reward for maintaining speed limit
        elif int(self.output['va'])== int(speedlimit):
            reward += r_maintainspeedlimit
            
        return reward
        
    def step(self, action):
        #reset flags
        self.done = False
        self.running = False
        self.collision = False
        self.lane_change = False

        veh_ids = traci.vehicle.getIDList()
        for veh_id in veh_ids:
            if veh_id != "Auto":
                traci.vehicle.setLaneChangeMode(veh_id,0b010001010101)
            else:
                traci.vehicle.setLaneChangeMode("Auto",0x0)
                traci.vehicle.setSpeedMode("Auto",0x1F)

        #left lane change
        if action == 1:
            self.change_left()
            # self.acc_counter = 0
            # self.decel_counter = 0
        #right lane change
        elif action == 2:
            self.change_right()
            # self.acc_counter = 0
            # self.decel_counter = 0
        # increase velocity
        elif action == 3:
            self.acc_counter +=1
            self.decel_counter = 0
            self.speed_up()
        # decrease velocity
        elif action == 4:
            self.decel_counter +=1
            self.acc_counter = 0
            self.speed_down()
        # idle
        else:
            # self.acc_counter = 0
            # self.decel_counter = 0
            self.idle()
        			
        #update scenario
        traci.simulationStep()
        #get scenario informations
        veh_ids = traci.vehicle.getIDList()
        for veh_id in veh_ids:
            traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_LENGTH, tc.VAR_LANE_ID, tc.VAR_ACCELERATION ))
        vehs = traci.vehicle.getAllSubscriptionResults()
        # vehs = traci.vehicle.getAllSubscriptionResults()

        self.num_vehs = len(vehs)
        sort_vehs=[]
        if 'Auto' in vehs.keys():
            #sort vehicles in simulation based on position
            sort_vehs = sort_nesteddict(vehs, 66)
            #GET AUTO INFORMATION
            self.auto_idx = list(sort_vehs.keys()).index('Auto')
            auto = list(sort_vehs.items())[self.auto_idx]
            #get auto position
            self.auto_pos = auto[1][tc.VAR_POSITION][0]
            #get auto length
            self.auto_length = auto[1][tc.VAR_LENGTH]
            #UPDATE AUTO INFORMATION
            #update auto speed
            self.output['va'] = auto[1][tc.VAR_SPEED]
            #update auto lane
            if auto[1][tc.VAR_LANE_ID] == "Lane_0":
                self.output['L'] = 0
            elif auto[1][tc.VAR_LANE_ID] == "Lane_1":
                self.output['L'] = 1
            elif auto[1][tc.VAR_LANE_ID] == "Lane_2":
                self.output['L'] = 2
            #update auto acceleration
            self.output['vacc'] = auto[1][tc.VAR_ACCELERATION]
            if self.output['vacc'] == 0:
                self.acc_counter = 0
                self.decel_counter = 0
            if abs(self.output['vacc']-self.next_acc) > TARGET_ACC_DIFF_LIMIT:
                self.target_spd = self.output['va']
                self.next_acc = 0
            
            #UPDATE TARGET INFORMATION
            #update leading vehicles
            self.find_forward_vehs(sort_vehs)
            #update trailing vehicles
            self.find_trailing_vehs(sort_vehs)

            #get rewards
            self.rewards = self.reward()
            self.sort_vehs = sort_vehs
            self.running = True
                
        
        self.output_prev = self.output.copy()
        
        return self.output, self.rewards, self.done, self.running
    
    def reset(self):
        self.close()
        self.seed +=1
        sumoCmd = [sumoBinary, "-c", sumoconfig, "--seed", str(self.seed), "--random="+ self.random_flag]
        traci.isLibsumo()
        traci.start(sumoCmd)
        self.output = dict(zip(output_keys,output_values))
        self.output_prev = self.output.copy()
        self.num_vehs = 0
        self.auto_idx = 0
        self.auto_pos = 0
        self.auto_length = 0
        self.maxspeed = traci.vehicletype.getMaxSpeed("Auto")
        self.maxspeed_slow = traci.vehicletype.getMaxSpeed("SlowCar")
        self.rewards=0
        self.test = True
        self.sort_vehs = []
        self.target_spd=5
        # traci.vehicle.setSpeedMode("Auto",0x1F)
        # traci.vehicle.setLaneChangeMode("Auto",0x0)
        self.ACTION_SPACE_SIZE = 5
        self.SPEED_LIMIT = speedlimit
        self.MAX_DISTANCE = max_dist
        self.default = self.output
        self.acc = 1.26
        self.decel = - 0.63
        self.next_acc = 0
        self.acc_counter = 0
        self.decel_counter = 0
        self.lane_change = False
        self.collision = False
        self.running = False
        self.done = False

        return self.output
        
    def close(self):
        traci.close()
    
            
            
            
        
        
