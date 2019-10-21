# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def helper(s, fringe):
    """
    :param s: search problem
    :param fringe: a (usually empty) queue or a stack
    :return:
    """
    searched = []

    #
    fringe.push([(
        s.getStartState(),  # state
        "stop",  # action
        0  # cost
    )
    ])

    while fringe.isEmpty() == False:
        path = fringe.pop()  # path

        current = path[len(path) - 1]  # current node/most recent node
        current = current[0]

        if s.isGoalState(current):
            return [i[1] for i in path][1:]

        if current not in searched:
            searched.append(current)  # the current state has been searched

            for child in s.getSuccessors(current):
                if child[0] not in searched:
                    # the path of child is the parent path and the child node
                    child_path = path[:]
                    child_path.append(child)
                    fringe.push(child_path)  # add child path to the fringe

    # return empty list if failed
    return []


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    dfs_stack = util.Stack()
    solution = helper(problem, dfs_stack)

    return solution

    # util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # total cost does not matter
    bfs_queue = util.Queue()
    solution = helper(problem, bfs_queue)

    return solution

    # util.raiseNotDefined()


"""
def backwardCost(problem, l):
    return problem.getCostOfActions([i[1] for i in l][1:])
"""


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    total_cost = lambda l:problem.getCostOfActions([i[1] for i in l][1:])
    ucs_priority_queue = util.PriorityQueueWithFunction(total_cost)
    solution = helper(problem, ucs_priority_queue)

    return solution

    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fx = lambda l: problem.getCostOfActions([i[1] for i in l][1:]) + heuristic(l[-1][0], problem)
    a_priority_queue = util.PriorityQueueWithFunction(fx)
    solution = helper(problem, a_priority_queue)

    return solution
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
