# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def __init__(self):
        self.prevPositions = []

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        if successorGameState.isWin():
            # successor state is win state -> return score > 1000
            return 1001

        food_list = newFood.asList()
        food_distances = [util.manhattanDistance(food, newPos) for food in food_list]
        closest_food = min(food_distances) * 3

        score = successorGameState.getScore()  # initial score
        ghost_distance = util.manhattanDistance(currentGameState.getGhostPosition(1), newPos)
        score += ghost_distance

        if currentGameState.getNumFood() > successorGameState.getNumFood():
            score += 100

        penalty = 10  # set penalty
        if action == Directions.STOP:
            score -= penalty
        score -= closest_food

        if successorGameState.getPacmanPosition() in currentGameState.getCapsules():
            score += 120
        """
        food = newFood.asList()

        food_distances = [manhattanDistance(x, newPos) for x in food]
        closest_food = min(food_distances)

        ghost_distances = [manhattanDistance(x.getPosition, newPos) for x in newGhostStates]
        closest_ghost_distances = min(ghost_distances)

        ghost_current_distances = [manhattanDistance(i, newPos) for i in ghost_distances]
        
        score = successorGameState.getScore() - currentGameState.getScore()
        if action == Directions.STOP:
            score -= penalty
        """
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def fminmax (self, gameState, agentIndex, depth):
        agents = gameState.getNumAgents()
        if agentIndex == agents:
            if depth == self.depth:
                return self.evaluationFunction(gameState)
            else:
                return self.fminmax(gameState, 0, depth + 1)
        else:
            legal_actions = gameState.getLegalActions(agentIndex)
            if len(legal_actions) == 0:
                return self.evaluationFunction(gameState)

            successors = [self.fminmax(gameState.generateSuccessor(agentIndex, i), agentIndex + 1, depth) for i in legal_actions]
            if agentIndex == 0:
                return max(successors)
            else:
                return min(successors)

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # init variables
        """
        ghosts = gameState.getNumAgents() - 1  # everyone but pacman
        score = -1000
        action = Directions.STOP  # worST
        tally = 0
        for something in gameState.getLegalActions():
            next_something = gameState.generateSuccessor(0, something)
            next_score = self.fmin(next_something, self.depth, 1, ghosts)
            if next_score > score:
                score = next_score
                action = something
            tally += 1
        
            compare = self.fmin(next_something, self.depth, 1, ghosts)
            if score <= compare:
                # print("ARRIVED") -- it is not coming here
                score = compare
                action = something
           
        """
        legal_action_0 = gameState.getLegalActions(0)
        return max(legal_action_0, key=lambda x: self.fminmax(gameState.generateSuccessor(0, x), 1, 1))


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def fmin(self, gameState, agentIndex, depth, alpha, beta):
        ghosts = gameState.getNumAgents()
        if agentIndex == ghosts:
            return self.fmax(gameState, 0, depth + 1, alpha, beta)

        temp = None
        legal_actions = gameState.getLegalActions(agentIndex)
        for action in legal_actions:
            next = gameState.generateSuccessor(agentIndex, action)
            next_func = self.fmin(next, agentIndex + 1, depth, alpha, beta)

            if temp is None:
                temp = next_func
            else:
                temp = min(temp, next_func)
            if alpha is not None and temp < alpha:
                return temp
            if beta is None:
                beta = temp
            else:
                beta = min(beta, temp)
        if temp is not None:
            return temp
        else:
            return self.evaluationFunction(gameState)

    def fmax(self, gameState, agentIndex, depth, alpha, beta):
        if depth > self.depth:
            return self.evaluationFunction(gameState)
        temp = 0
        legal_actions = gameState.getLegalActions(agentIndex)
        for action in legal_actions:
            next = gameState.generateSuccessor(agentIndex, action)
            next_func = self.fmin(next, agentIndex + 1, depth, alpha, beta)

            temp = max(temp, next_func)

            if beta is not None and temp > beta:
                return temp
            alpha = max(alpha, temp)

        if temp is not None:
            return temp
        else:
            return self.evaluationFunction(gameState)

    def listy(self, l):
        return [x for x in l if x != "Stop"]

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        scores = []

        def helper(gameState, iterCount, alpha, beta):
            ghosts = gameState.getNumAgents()
            compare = self.depth * ghosts
            if gameState.isWin() or gameState.isLose() or iterCount >= compare:
                return self.evaluationFunction(gameState)
            ghost_min = iterCount % ghosts
            if ghost_min != 0:
                result = 1e10
                legal_actions = gameState.getLegalActions(ghost_min)
                for action in self.listy(legal_actions):
                    successors = gameState.generateSuccessor(ghost_min, action)
                    result = min(result, helper(successors, iterCount + 1, alpha, beta))
                    beta = min(beta, result)
                    if beta < alpha:
                        break
                return result
            else:
                result = -1e10
                legal_actions = gameState.getLegalActions(ghost_min)
                for action in self.listy(legal_actions):
                    successors = gameState.generateSuccessor(ghost_min, action)
                    result = max(result, helper(successors, iterCount + 1, alpha, beta))
                    alpha = max(alpha, result)
                    if iterCount == 0:
                        scores.append(result)
                    if beta < alpha:
                        break
                return result

        results = helper(gameState, 0, -1e20, 1e20)
        legal = gameState.getLegalActions(0)
        legal_list = self.listy(legal)
        return legal_list[scores.index(max(scores))]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def listy(self, l):
        return [x for x in l if x != "Stop"]

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        scores = []

        def helper(gameState, iterCount):
            ghosts = gameState.getNumAgents()
            compare = self.depth * ghosts
            if gameState.isWin() or gameState.isLose() or iterCount >= compare:
                return self.evaluationFunction(gameState)

            ghost_min = iterCount % ghosts
            if ghost_min != 0:
                suc_scores = []
                legal_actions = gameState.getLegalActions(ghost_min)
                for action in self.listy(legal_actions):
                    successor = gameState.generateSuccessor(ghost_min, action)
                    result = helper(successor, iterCount + 1)
                    suc_scores.append(result)
                average = 0
                for score in suc_scores:
                    average += float(score)/len(suc_scores)
                return average
            else:
                result = -1e10
                legal_actions = gameState.getLegalActions(ghost_min)
                for action in self.listy(legal_actions):
                    successor = gameState.generateSuccessor(ghost_min, action)
                    result = max(result, helper(successor, iterCount + 1))
                    if iterCount == 0:
                        scores.append(result)
            return result

        r = helper(gameState, 0)
        legal = gameState.getLegalActions(0)
        legal_list = self.listy(legal)
        return legal_list[scores.index(max(scores))]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Score for each ghosts, food, and capsules and then add it up
    to the game score
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newCapsules = currentGameState.getCapsules()
    foodList = currentGameState.getFood().asList()

    ghosts_score = 0
    food_score = 0
    capsules_score = 0

    for ghost in currentGameState.getGhostStates():
        ghosts_distance = manhattanDistance(newPos, ghost.getPosition())
        if ghost.scaredTimer > 0: # lol i got errors for 2 days bc i wrote sacred Timer
            power = pow(max(8 - ghosts_distance, 0), 2)
            ghosts_score += power
        else:
            power = pow(max(7 - ghosts_distance, 0), 2)
            ghosts_score -= power

    food_distances = []
    for food in foodList:
        dis = 1/manhattanDistance(newPos, food)
        food_distances.append(dis)
    if len(food_distances) > 0:
        food_score = max(food_distances)

    all_capsules = []
    for c in newCapsules:
        power = 50/manhattanDistance(newPos, c)
        all_capsules.append(power)
    if len(all_capsules) > 0:
        capsules_score = max(all_capsules)

    return currentGameState.getScore() + ghosts_score + food_score + capsules_score


    """
    newPosition = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhost = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    foodList = newFood.asList()

    ghost_distances = [manhattanDistance(newPosition, i.getPosition()) for i in newGhost]
    closest_ghost = min(ghost_distances)

    closest_cap = 0
    cap_score = 100
    ghost_score = -500
    food_score = 0

    if newCapsules:
        capsules = [manhattanDistance(newPosition, i) for i in newCapsules]
        closest_cap = min(capsules)
    if closest_cap:
        cap_score = -3/closest_cap
    if closest_ghost:
        ghost_score = -2/closest_ghost
    if foodList:
        foods = [manhattanDistance(newPosition, i) for i in foodList]
        food_score = min(foods)

    score = -2 * food_score + ghost_score - 10 * len(foodList) + cap_score
    return score
    """

# Abbreviation
better = betterEvaluationFunction
