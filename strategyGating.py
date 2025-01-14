#!/usr/bin/env python
import collections
import os

from radarGuidance import *
from wallFollower import *

import random #used for the random choice of a strategy
import sys
import numpy as np
import math
import time



#--------------------------------------
# Position of the goal:
goalx = 300
goaly = 450
# Initial position of the robot:
initx = 300
inity = 35
# strategy choice related stuff:
choice = -1
choice_tm1 = -1
tLastChoice = 0
rew = 0

i2name=['wallFollower','radarGuidance']

# Parameters of State building:
# threshold for wall consideration
th_neglectedWall = 35
# threshold to consider that we are too close to a wall
# and a punishment should be delivered
th_obstacleTooClose = 13
# angular limits used to define states
angleLMin = 0
angleLMax = 55

angleFMin=56
angleFMax=143

angleRMin=144
angleRMax=199

# Q-learning related stuff:
# definition of states at time t and t-1
S_t = ''
S_tm1 = ''

#--------------------------------------
# the function that selects which controller (radarGuidance or wallFollower) to use
# sets the global variable "choice" to 0 (wallFollower) or 1 (radarGuidance)
# * arbitrationMethod: how to select? 'random','randPersist','qlearning'
global lastChooseTime
lastChooseTime = 0
Qtable = collections.defaultdict(lambda : np.zeros(2))


def strategyGating(arbitrationMethod,verbose=True):
  global choice
  global choice_tm1
  global tLastChoice
  global rew
  global lastChooseTime
  # The chosen gating strategy is to be coded here:
  #------------------------------------------------
  # print(arbitrationMethod)
  if arbitrationMethod=='random':
    choice = random.randrange(2)
  #------------------------------------------------
  elif arbitrationMethod=='randomPersist':
    if time.time() - lastChooseTime >= 2:
      # print(lastChooseTime)
      lastChooseTime = time.time()
      choice = random.randrange(2)
  #------------------------------------------------
  elif arbitrationMethod=='qlearning':
    gamma = 0.95
    alpha = 0.4

    if time.time() - lastChooseTime >= 2:
      # make choose----------------
      beta = 4
      p = beta * np.array(Qtable[S_t])
      p = np.exp(p) / sum(np.exp(p))
      choice = 1
      print(p[0])
      if np.random.rand() < p[0]:
        choice = 0
      #---------------------------
      lastChooseTime = time.time()
      # update Qtable--------------
      delta = rew + gamma * np.max(Qtable[S_t]) - Qtable[S_tm1][choice_tm1]
      Qtable[S_tm1][choice_tm1] += alpha * delta
      #---------------------------
      choice_tm1 = choice
      rew = 0
  #------------------------------------------------
  else:
    print(arbitrationMethod+' unknown.')
    exit()

  if verbose:
    print("strategyGating: Active Module: "+i2name[choice])

#--------------------------------------
def buildStateFromSensors(laserRanges,radar,dist2goal):
  S   = ''
  # determine if obstacle on the left:
  wall='0'
  if min(laserRanges[angleLMin:angleLMax]) < th_neglectedWall:
    wall ='1'
  S += wall
  # determine if obstacle in front:
  wall='0'
  if min(laserRanges[angleFMin:angleFMax]) < th_neglectedWall:
    wall ='1'
    #print("Mur Devant")
  S += wall
  # determine if obstacle on the right:
  wall='0'
  if min(laserRanges[angleRMin:angleRMax]) < th_neglectedWall:
    wall ='1'
  S += wall

  S += str(radar)

  if dist2goal < 125:
    S+='0'
  elif dist2goal < 250:
    S+='1'
  else:
    S+='2'
  #print('buildStateFromSensors: State: '+S)

  return S

#--------------------------------------
def main(argv):
  global S_t
  global S_tm1
  global rew
  settings = Settings('worlds/entonnoir.xml')

  env_map = settings.map()
  robot = settings.robot()

  d = Display(env_map, robot)

  method = argv[1]
  # experiment related stuff
  startT = time.time()
  trial = 0
  nbTrials = 40
  trialDuration = np.zeros((nbTrials))
  list_pos = []
  i = 0
  while trial<nbTrials:
    # update the display
    #-------------------------------------
    d.update()
    # get position data from the simulation
    pos = robot.get_pos()
    next_time = time.time()
    #-------------------------------------
    pos = robot.get_pos()
    prev_time = 0
    # print("##########\nStep "+str(i)+" robot pos: x = "+str(int(pos.x()))+" y = "+str(int(pos.y()))+" theta = "+str(int(pos.theta()/math.pi*180.)))
    if next_time - prev_time >= 1 or trial == 0:
    	list_pos.append([pos.x(), pos.y(), pos.theta() / math.pi * 180.])
    	prev_time = next_time
    # has the robot found the reward ?
    #------------------------------------
    dist2goal = math.sqrt((pos.x()-goalx)**2+(pos.y()-goaly)**2)
    # if so, teleport it to initial position, store trial duration, set reward to 1:
    if (dist2goal<20): # 30
      print('***** REWARD REACHED *****')
      pos.set_x(initx)
      pos.set_y(inity)
      robot.set_pos(pos) # format ?
      # and store information about the duration of the finishing trial:
      currT = time.time()
      trialDuration[trial] = currT - startT
      startT = currT
      print("Trial "+str(trial)+" duration:"+str(trialDuration[trial]))
      if method == 'qlearning':
        f = open('log/' + str(startT) + '-Trial-' + str(trial) + '-Positions-' + method + '.txt', 'w+')
        f.close()
        np.savetxt('log/' + str(startT) + '-Trial-' + str(trial) + '-Positions-' + method + '.txt', np.array(list_pos))
      trial +=1
      rew = 1
      list_pos = []

    # get the sensor inputs:
    #------------------------------------
    lasers = robot.get_laser_scanners()[0].get_lasers()
    laserRanges = []
    for l in lasers:
      laserRanges.append(l.get_dist())

    radar = robot.get_radars()[0].get_activated_slice()

    bumperL = robot.get_left_bumper()
    bumperR = robot.get_right_bumper()


    # 2) has the robot bumped into a wall ?
    #------------------------------------
    if bumperR or bumperL or min(laserRanges[angleFMin:angleFMax]) < th_obstacleTooClose:
      rew = -1
      print("***** BING! ***** "+i2name[choice])

    # 3) build the state, that will be used by learning, from the sensory data
    #------------------------------------
    S_tm1 = S_t
    S_t = buildStateFromSensors(laserRanges,radar, dist2goal)

    #------------------------------------
    strategyGating(method,verbose=False)
    if choice==0:
      v = wallFollower(laserRanges,verbose=False)
    else:
      v = radarGuidance(laserRanges,bumperL,bumperR,radar,verbose=False)

    i+=1
    robot.move(v[0], v[1], env_map)
    time.sleep(0.01)

  # When the experiment is over:


  np.savetxt('log/'+str(startT)+'-TrialDurations-'+method+'.txt',trialDuration)

  res = dict()
  aux = {}
  for key in Qtable:
    print('key:', Qtable[key])
  for key in Qtable:
    if key in ['00002','00072','00000','00070','11101','11171']:
      aux[key] = Qtable[key]
    res[key] = Qtable[key]
    print('key:', Qtable[key])
  print(aux)
  if method == 'qlearning':
    np.save('log/' + str(startT) + '-TrialQvalues-'+method+'.npy', res)

#--------------------------------------

if __name__ == '__main__':
  random.seed()
  main(sys.argv)
