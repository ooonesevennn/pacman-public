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
from game import Directions, Actions
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

        self.epsilon = 0.0 #exploration prob
        self.alpha = 0.2 #learning rate
        self.discountRate = 0.8
        self.offensiveWeights = {'closest-food': -1, 
                                        'bias': 1, 
                                        '#-of-ghosts-1-step-away': -10, 
                                        'successorScore': 100, 
                                        'eats-food': 10}
        self.defensiveWeights = {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}	
        self.escapeWeights = {'onDefense': 1000, 'enemyDistance': 30, 'stop': -100, 'distanceToHome': -20}
        
        """
        Open weights file if it exists, otherwise start with empty weights.
        NEEDS TO BE CHANGED BEFORE SUBMISSION

        """
        if os.path.exists('offensiveWeights.txt'):
            with open('offensiveWeights.txt', "r") as file:
                self.offensiveWeights = eval(file.read())
        
        if os.path.exists('defensiveWeights.txt'):
            with open('offensiveWeights.txt', "r") as file:
                self.defensiveWeights = eval(file.read())
        
        if os.path.exists('escapeWeights.txt'):
            with open('escapeWeights.txt', "r") as file:
                self.escapeWeights = eval(file.read())
    
    #------------------------------- Q-learning Functions -------------------------------

    """
    Iterate through all features (closest food, bias, ghost dist),
    multiply each of the features' value to the feature's weight,
    and return the sum of all these values to get the q-value.
    """
    def getQValue(self, gameState, action):
        #############
        # Implement your code to calculate and return Q value
        #############

        q_value = 0
        return q_value

    """
    Iterate through all q-values that we get from all
    possible actions, and return the highest q-value
    """
    def getValue(self, gameState):
        legalActions = gameState.getLegalActions(self.index)
        if len(legalActions) == 0:
                return 0.0
        else:
                #############
                # Implement your return max Q value from all legalActions for a given state
                #############
                maxQvalue = 0
                return maxQvalue

    """
    Iterate through all q-values that we get from all
    possible actions, and return the action associated
    with the highest q-value.
    """
    def getPolicy(self, gameState):
        values = []
        legalActions = gameState.getLegalActions(self.index)
        legalActions.remove(Directions.STOP)
        if len(legalActions) == 0:
                return None
        else:
                for action in legalActions:
                        self.updateWeights(gameState, action)
                        values.append((self.getQValue(gameState, action), action))
        return max(values)[1]
    
    """
    Iterate through all features and for each feature, update
    its weight values using the following formula:
    w(i) = w(i) + alpha((reward + discount*value(nextState)) - Q(s,a)) * f(i)(s,a)
    """
    def updateWeights(self, gameState, action):
        features = self.getFeatures(gameState, action)
        nextState = self.getSuccessor(gameState, action)

        # Calculate the reward. NEEDS WORK
        reward = nextState.getScore() - gameState.getScore()

        for feature in features:
            ####################
            # Impletement your codes to perform Approximate Q-learning update on each weight in self.weights.
            ####################
            pass 
    
    #------------------------------- PDDL Functions ------------------------------- 
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

    #------------------------------- Low Level Action Functions -------------------------------

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
    
    """
    Calculate probability of 0.1.
    If probability is < 0.1, then choose a random action from
    a list of legal actions.
    Otherwise use the policy defined above to get an action.
    """
    def chooseOffensiveQLearningAction(self, gameState):
        # This function is not using at this stage. you need to complete this function and replace the old chooseOffensiveAction function.
        # Pick Action
        legalActions = gameState.getLegalActions(self.index)
        action = None

        if len(legalActions) != 0:
                ############
                # At here, current code only returns actions through calculate Q value.
                # Change the codes here to have a probability of self.epsilon to return random action.
                ###########

                action = self.getPolicy(gameState)
        return action
    
    def getOffensiveEvaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getOffensiveFeatures(gameState, action)
        weights = self.getOffensiveWeights(gameState, action)
        return features * weights
    
    def getOffensiveFeatures(self, gameState, action):
        food = self.getFood(gameState) 
        walls = gameState.getWalls()
        ghosts = []
        opAgents = CaptureAgent.getOpponents(self, gameState)
        # Get ghost locations and states if observable
        if opAgents:
                for opponent in opAgents:
                        opPos = gameState.getAgentPosition(opponent)
                        opIsPacman = gameState.getAgentState(opponent).isPacman
                        if opPos and not opIsPacman: 
                                ghosts.append(opPos)
        
        # Initialize features
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # Successor Score
        features['successorScore'] = self.getScore(successor)

        # Bias
        features["bias"] = 1.0
        
        # compute the location of pacman after he takes the action
        x, y = gameState.getAgentPosition(self.index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        
        # Number of Ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
                features["eats-food"] = 1.0

        # Number of Ghosts scared
        #features['#-of-scared-ghosts'] = sum(gameState.getAgentState(opponent).scaredTimer != 0 for opponent in opAgents)
        
        # Closest food
        dist = self.closestFood((next_x, next_y), food, walls)
        if dist is not None:
                # make the distance a number less than one otherwise the update
                # will diverge wildly
                features["closest-food"] = float(dist) / (walls.width * walls.height) 

        # Normalize and return
        features.divideAll(10.0)
        return features

    def getOffensiveWeights(self, gameState, action):
        return self.offensiveWeights
    
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
        return self.escapeWeights
    
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
        return self.defensiveWeights
    
    def closestFood(self, pos, food, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None
    
    def final(self, gameState):
        print(self.weights)
        file = open('offensiveWeights.txt', 'w')
        file.write(str(self.weights))
        file.close()

        file = open('defensiveWeights.txt', 'w')
        file.write(str(self.weights))
        file.close()

        file = open('escapeWeights.txt', 'w')
        file.write(str(self.weights))
        file.close()
