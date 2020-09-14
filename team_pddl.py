# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, os
from game import Directions
import game
from util import nearestPoint

from pddlstream.algorithms.search import Pddl_Domain
from pddlstream.utils import read
from pddlstream.language.constants import And, Equal,ForAll, print_solution, Type,TOTAL_COST,EQ, Not

#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first = 'MixedAgent', second = 'MixedAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    values = [self.evaluate(gameState, a) for a in actions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())
    
    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class MixedAgent(ReflexCaptureAgent):
  """
  This is an agent that use pddl to guide the high level actions of Pacman
  """
  domain_pddl = Pddl_Domain('pacman_bool.pddl')

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def get_pddl_state(self,gameState):
    new_state = {
    "enemy_at_home" : False,
    "enemy_around" : False,
    "at_home" : False,
    "at_enemy_land" : False,
    "food_in_backpack" : False,
    "food_at_playground" : False,
    }

    foodLeft = self.getFood(gameState).asList()
    new_state["food_at_playground"] = len(foodLeft) > 0

    state = gameState.getAgentState(self.index)
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

    if state.numCarrying>0:
      new_state["food_in_backpack"]=True
    if state.isPacman:
      new_state["at_enemy_land"]= True
    else:
      new_state["at_home"] = True
    
    myPos = state.getPosition()
    for enemy in enemies:
      enemy_position = enemy.getPosition()
      if enemy_position != None and self.getMazeDistance(myPos, enemy_position) <= 5:
        new_state["enemy_around"] = True
      if enemy.isPacman:
        new_state["enemy_at_home"] = True
    return new_state
  
  def getHighLevelAction(self,gameState):
    pddl_state = self.get_pddl_state(gameState)

    objects = []
    init = []
    for key, item in pddl_state.items():
      if item:
        init += [(key,)]
    
    temp_goal = [Not(("food_at_playground",)),Not(("food_in_backpack",)),Not(("enemy_at_home",))]
    goal =  And(*temp_goal)

    plan, cost = self.domain_pddl.solve(objects,init, goal,planner="ff-astar",unit_costs=False,debug=False)
    if len(plan)==0:
      return None
    print("Agent: ", self.index, plan[0].name)
    return plan[0].name
  
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """

    # get high level action from pddl module
    highLevelAction = self.getHighLevelAction(gameState)

    # get detailed actions based on high level actions. 
    if highLevelAction == "go_to_enemy_playground" or highLevelAction == "eat_food":
      action = self.getOffensiveAction(gameState)
    elif highLevelAction == "go_home" or highLevelAction == "unpack_food":
      action = self.getEscapeAction(gameState)
    else:
      action = self.getDefensiveAction(gameState)
    return action
  
  def getOffensiveAction(self, gameState):
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.getOffensiveEvaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())
    
    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)
  
  def getOffensiveEvaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getOffensiveFeatures(gameState, action)
    weights = self.getOffensiveWeights(gameState, action)
    return features * weights
  
  def getOffensiveFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()  
    myPos = successor.getAgentState(self.index).getPosition()


    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['distanceToEnemy'] = min(dists)
    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getOffensiveWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}
  
  def getEscapeAction(self, gameState):
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.getEscapeEvaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def getEscapeEvaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getEscapeFeatures(gameState, action)
    weights = self.getEscapeWeights(gameState, action)
    return features * weights

  def getEscapeFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    enemiesAround = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(enemiesAround) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemiesAround]
      features['enemyDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    features["distanceToHome"] = self.getMazeDistance(myPos,myState.start.getPosition())

    return features

  def getEscapeWeights(self, gameState, action):
    return {'onDefense': 1000, 'enemyDistance': 30, 'stop': -100, 'distanceToHome': -20}
  
  def getDefensiveAction(self, gameState):
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.getDefensiveEvaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def getDefensiveEvaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getDefensiveFeatures(gameState, action)
    weights = self.getDefensiveWeights(gameState, action)
    return features * weights

  def getDefensiveFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getDefensiveWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
