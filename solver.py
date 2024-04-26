import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
global posWalls, posGoals
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 

    # print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     

    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = [[0]] 
    temp = []
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])
    return temp

def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState) # get initial positions of boxes
    beginPlayer = PosOfPlayer(gameState) # get initial positions of player

    startState = (beginPlayer, beginBox) # Create initial state to store positions of player and box
    frontier = collections.deque([[startState]]) # Create a frontier queue, with the initial point being the state of the game.
    actions = collections.deque([[0]]) # store actions
    exploredSet = set() # Create a set to store the explored states
    temp = [] # create an empty list to store the resulting actions
    
    #BFS algorithm:
    while frontier: # While there are nodes to explore
        node = frontier.popleft() # Get the leftside state in the frontier and remove it
        node_action = actions.popleft() # Get the leftside action (path) in the actions list and remove it
        if isEndState(node[-1][-1]): # If the current node is the goal state
            temp += node_action[1:] # Append all actions into temp list
            break # break out of the loop
        if node[-1] not in exploredSet: # check if the current node has not been visited.
            exploredSet.add(node[-1]) # Add the current node to the set of visited nodes.
            for action in legalActions(node[-1][0], node[-1][1]): # Loop through the set of legal action, for example (UP, DOWN, LEFT, RIGHT)
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # update positions of player and boxes using the action
                if isFailed(newPosBox): # if the resulting box positions lead to a failed state
                    continue # Skip and go to the next action
                frontier.append(node + [(newPosPlayer, newPosBox)]) # Add new position to frontier queue
                actions.append(node_action + [action[-1]]) # Add new action to the actions
    return temp # Return the actions lead to the goal state

def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])

def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    beginBox = PosOfBoxes(gameState) # get initial positions of boxes
    beginPlayer = PosOfPlayer(gameState) # get initial positions of player

    startState = (beginPlayer, beginBox) # Create initial state to store positions of player and box
    frontier = PriorityQueue() # create a priority queue for the frontier
    frontier.push([startState], 0) # the initial point being the state of the game
    exploredSet = set() # create a set to store the explored states
    actions = PriorityQueue() # create a priority queue for the actions 
    actions.push([0], 0) # add the initial action to the actions queue with priority 0 (the cost of first node is 0)
    temp = [] # create an empty list to store the resulting actions
    
    num_explored_nodes = 0 # Initialize counter variable to count number of explored nodes
    
    #UCS algorithm:
    while not frontier.isEmpty(): # While there are nodes to explore
        node = frontier.pop() # Pop the node from the frontier with the lowest priority (last position) and remove it
        node_action = actions.pop() # Pop the last action in actions queue and remove it
        if isEndState(node[-1][-1]):  # If the current node is the goal state
            temp += node_action[1:] # Append all actions into temp list, except the first (0).
            break # break out of the loop
        if node[-1] not in exploredSet: # If the current node has not been visited
            num_explored_nodes = num_explored_nodes + 1 # Increase the explored node count variable by 1
            
            exploredSet.add(node[-1]) # Add the current node to the set of visited nodes.
            Cost_of_Action = cost(node_action[1:]) # Calculate the cost of actions (the initial position to the current node) when not pushing
            for action in legalActions(node[-1][0], node[-1][1]): # Loop through the set of legal action (UP, DOWN, LEFT, RIGHT)
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # Update the new player and box positions.
                if isFailed(newPosBox): # If the new box position is a fail state.
                    continue # Skip and go to the next action
                frontier.push(node + [(newPosPlayer, newPosBox)], Cost_of_Action) # Add the new node with the new player and box positions to the frontier with the priority of the cost.
                actions.push(node_action + [action[-1]], Cost_of_Action) # Add the new node to the actions with the priority of the cost.
    
    print('Number of explored nodes:', num_explored_nodes) #print number of explored nodes
    return temp # Return the actions lead to the goal state and number of explored nodes

def heuristic(posPlayer, posBox):
    # print(posPlayer, posBox)
    """A heuristic function to calculate the overall distance between the else boxes and the else goals"""
    distance = 0 # Initialize distance variable to store the total distance
    completes = set(posGoals) & set(posBox) # Find the positions where boxes are already on goals to identify the positions where boxes have already reached goals.
    
    # 'difference' is used to find elements in one set but not in the other, effectively excluding completed positions.
    sortposBox = list(set(posBox).difference(completes)) # Remove the completed positions (where box is already on goal) 
    sortposGoals = list(set(posGoals).difference(completes)) # from both the box and goal positions.
    for i in range(len(sortposBox)): # Iterate through the remaining box positions.
        # Calculate the Manhattan distance between the box and its corresponding goal and add the this distance between each box and its corresponding goal.
        distance += (abs(sortposBox[i][0] - sortposGoals[i][0])) + (abs(sortposBox[i][1] - sortposGoals[i][1]))
    return distance # Return the total calculated distance.


def heuristic_new(posPlayer, posBox):
    """A heuristic function to calculate the overall distance between the boxes and the goals"""
    distance = 0 # Initialize distance variable to store the total distance
    completes = set(posGoals) & set(posBox) # Find the positions where boxes are already on goals to identify the positions where boxes have already reached goals.
   
    # 'difference' is used to find elements in one set but not in the other, effectively excluding completed positions.
    sortposBox = list(set(posBox).difference(completes)) # Remove the completed positions (where box is already on goal)
    sortposGoals = list(set(posGoals).difference(completes)) # from both the box and goal positions.
    for i in range(len(sortposBox)): # Iterate through the remaining box positions.
        
        # Calculate the Euclidean distance between the box and its corresponding goal.
        # Euclidean distance is the straight-line distance between two points in a plane.
        distance += np.sqrt((sortposBox[i][0] - sortposGoals[i][0])**2 + (sortposBox[i][1] - sortposGoals[i][1])**2)
    return distance # Return the total calculated distance.

def aStarSearch(gameState):
    """ A* uses the sum of the actual costs passed and the remaining cost estimate (heuristic) to decide which state should be expanded next
    (search for a potentially more optimal path), while UCS searches based on the actual cost of each step."""
    
    beginBox = PosOfBoxes(gameState) # Get the initial positions of boxes 
    beginPlayer = PosOfPlayer(gameState) # Get the initial positions of player
    temp = [] # Initialize a list to store the actions to reach the goal state
    start_state = (beginPlayer, beginBox) # Initialize the start state with the player and box positions
    frontier = PriorityQueue() # Initialize a priority queue to store states (nodes) based on their heuristic values
    frontier.push([start_state], heuristic(beginPlayer, beginBox)) # Push the start state into the frontier with its heuristic value as priority
    exploredSet = set() # Initialize an empty set to store explored states
    
    actions = PriorityQueue() # Initialize a priority queue to store actions along with their costs
    actions.push([0], heuristic(beginPlayer, start_state[1])) # Push the initial action sequence (empty) onto the actions queue with its heuristic value as priority
    
    num_explored_nodes = 0 # Initialize counter variable to count number of explored nodes

    while len(frontier.Heap) > 0: # While there are states in the frontier
        node = frontier.pop() # Pop the node from the frontier with the lowest priority (last position) and remove it
        node_action = actions.pop() # Pop the last action in actions queue and remove it
        
        if isEndState(node[-1][-1]): # This line checks if the popped state represents the goal state where all boxes are in their goal positions.
            temp += node_action[1:] # Append the actions to reach the goal state
            break # break out of the loop
        if node[-1] not in exploredSet: # If the current node has not been visited
            num_explored_nodes = num_explored_nodes + 1 #Increase the explored node count variable by 1
            
            exploredSet.add(node[-1]) # Add the current node to the set of visited nodes.
            Cost = cost(node_action[1:]) # Calculate the cost of actions (the initial position to the current node)
            for action in legalActions(node[-1][0], node[-1][1]): # Loop through the set of legal action (UP, DOWN, LEFT, RIGHT)
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # Update the new player and box positions.
                if isFailed(newPosBox): # If the new box position is a fail state.
                    continue # Skip and go to the next action
                Heuristic = heuristic(newPosPlayer, newPosBox) # Calculate the distance base on heuristic functions for the new state
                frontier.push(node + [(newPosPlayer, newPosBox)], Heuristic + Cost) # These lines push the new state and its total cost (estimated costs + actual cost) onto the frontier priority queue.
                actions.push(node_action + [action[-1]], Heuristic + Cost) # push the new action sequence and its total cost onto the actions priority queue.

    print('Number of explored nodes:', num_explored_nodes) #print number of explored nodes
    return temp #returns the sequence of actions leading to the goal state.

"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    
    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':        
        result = breadthFirstSearch(gameState)
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    elif method == 'astar':
        result = aStarSearch(gameState)        
    else:
        raise ValueError('Invalid method.')
    time_end = time.time()
    print('Runtime of %s: %.2f second.' %(method, time_end-time_start))
    print(result)
    print('Number of steps:', len(result))
    return result
