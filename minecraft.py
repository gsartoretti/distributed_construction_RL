import gym
from gym import spaces
import numpy as np
from collections import OrderedDict
from threading import Lock
import sys

'''
3D Grid Environment
    Observation: (position maps of current agent, other agents, blocks, sources, and plan)
        Position:   X, Y, Z  (+Y = up)
        View:       A box centered around the agent (limited view)
            block = -1
            block spawner = -2
            air = 0
            agent = 1 (agent_id in id_visible mode, agent_id is a positive integer)
            out of world range = -3

    Action space: (Tuple)
        agent_id: positive integer
        action: {0:NOP, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST, 5:PICK_NORTH,
        6:PICK_EAST, 7:PICK_SOUTH, 8:PICK_WEST, 9:PLACE_NORTH, 10:PLACE_EAST, 11:PLACE_SOUTH, 12:PLACE_WEST}
        NORTH: +Y, EAST: +X

    Reward: ACTION_COST for each action, REWARD for each block correctly placed
'''

'''
Rules (specifics):
A robot cannot walk on a block currently carried by an other robot.
A robot cannot place a block on an other robot or on a source.
A robot cannot walk over an other robot or a source.
A robot cannot be on the highest level of the simulation (world_shape[1]-1).
'''

PLAN_MAPS = [
## Simple 6x6 castle (1 block high) with 4 towers (3 blocks high)
    [[2, 0, 2], [2, 1, 2], [2, 2, 2], [2, 0, 3], [2, 0, 4], [2, 0, 5], [2, 0, 6], [7, 0, 2], [7, 1, 2], [7, 2, 2], [3, 0, 2], [3, 0, 7], [4, 0, 2], [4, 0, 7], [5, 0, 2], [5, 0, 7], [6, 0, 2], [6, 0, 7], [2, 0, 7], [2, 1, 7], [2, 2, 7], [7, 0, 3], [7, 0, 4], [7, 0, 5], [7, 0, 6], [7, 0, 7], [7, 1, 7], [7, 2, 7]],
## 4 1x3 towers only
    [[2, 0, 2], [2, 1, 2], [2, 2, 2], [7, 0, 2], [7, 1, 2], [7, 2, 2], [2, 0, 7], [2, 1, 7], [2, 2, 7], [7, 0, 7], [7, 1, 7], [7, 2, 7]],
## Pyramid centered and 2x2x3 high at the middle
    [[2, 0, 2], [2, 0, 3], [2, 0, 4], [2, 0, 5], [2, 0, 6], [2, 0, 7], [3, 0, 2], [3, 0, 3], [3, 0, 4], [3, 0, 5], [3, 0, 6], [3, 0, 7], [4, 0, 2], [4, 0, 3], [4, 0, 4], [4, 0, 5], [4, 0, 6], [4, 0, 7], [5, 0, 2], [5, 0, 3], [5, 0, 4], [5, 0, 5], [5, 0, 6], [5, 0, 7], [6, 0, 2], [6, 0, 3], [6, 0, 4], [6, 0, 5], [6, 0, 6], [6, 0, 7], [7, 0, 2], [7, 0, 3], [7, 0, 4], [7, 0, 5], [7, 0, 6], [7, 0, 7], [3, 1, 3], [3, 1, 4], [3, 1, 5], [3, 1, 6], [4, 1, 3], [4, 1, 4], [4, 1, 5], [4, 1, 6], [5, 1, 3], [5, 1, 4], [5, 1, 5], [5, 1, 6], [6, 1, 3], [6, 1, 4], [6, 1, 5], [6, 1, 6], [4, 2, 4], [4, 2, 5], [5, 2, 4], [5, 2, 5]],
## 1 big center cube (3x3x3)
    [[3, 0, 3], [3, 1, 3], [3, 2, 3], [3, 0, 4], [3, 1, 4], [3, 2, 4], [3, 0, 5], [3, 1, 5], [3, 2, 5], [4, 0, 3], [4, 1, 3], [4, 2, 3], [4, 0, 4], [4, 1, 4], [4, 2, 4], [4, 0, 5], [4, 1, 5], [4, 2, 5], [5, 0, 3], [5, 1, 3], [5, 2, 3], [5, 0, 4], [5, 1, 4], [5, 2, 4], [5, 0, 5], [5, 1, 5], [5, 2, 5]],
## 1 big cross wall (1x6x3 + 6x1x3)
    [[2, 0, 4], [2, 1, 4], [2, 2, 4], [3, 0, 4], [3, 1, 4], [3, 2, 4], [4, 0, 4], [4, 1, 4], [4, 2, 4], [5, 0, 4], [5, 1, 4], [5, 2, 4], [6, 0, 4], [6, 1, 4], [6, 2, 4], [7, 0, 4], [7, 1, 4], [7, 2, 4], [4, 0, 2], [4, 1, 2], [4, 2, 2], [4, 0, 3], [4, 1, 3], [4, 2, 3], [4, 0, 4], [4, 1, 4], [4, 2, 4], [4, 0, 5], [4, 1, 5], [4, 2, 5], [4, 0, 6], [4, 1, 6], [4, 2, 6], [4, 0, 7], [4, 1, 7], [4, 2, 7]],
## Center colulmn
    [[4, 0, 4], [4, 1, 4], [4, 2, 4], [4, 0, 5], [4, 1, 5], [4, 2, 5], [5, 0, 4], [5, 1, 4], [5, 2, 4], [5, 0, 5], [5, 1, 5], [5, 2, 5]]
]
SOURCES = [[0, 0, 0], [0, 0, 9], [9, 0, 0], [9, 0, 9]]

opposite_actions = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 9, 6: 10, 7: 11, 8: 12, 9: 5, 10: 6, 11: 7, 12: 8}
ACTION_COST, PLACE_REWARD = -0.02, +1.

BLOCK = np.array((210,105,30)) / 256.0
AIR = np.array((250,250,250)) / 256.0
PLAN_COLOR = np.array((250, 100, 100)) / 256.0
BLOCK_SPAWN = np.array((220,20,60)) / 256.0
AGENT = np.array((50,205,50)) / 256.0
OUT_BOUNDS = np.array((189,183,107)) / 256.0

class Grid3DState(object):
    '''
    3D Grid State.
    Implemented as a 3d numpy array.
        ground = -3
        block spawner = -2
        air = -1
        block = 0
        agent = positive integer (agent_id)
    '''
    def __init__(self, world0, num_agents=1):
        self.state = world0.copy()
        self.shape = np.array(world0.shape)
        self.num_agents = num_agents
        self.scanWorld()

    # Scan self.state for agents and load them into database
    def scanWorld(self):
        agents_list = []
        self.agents_pos = np.zeros((self.num_agents+1,3)) # x,y,z of each agent at start

        # list all agents
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    val = self.getBlock([i, j, k])
                    if val > 0:
                        assert val not in agents_list, 'ID conflict between agents'
                        assert type(val) is int or float, 'Non-integer agent ID'
                        val = int(val)
                        agents_list.append(val)
                        self.agents_pos[val] = [i,j,k]

        assert len(agents_list) == self.num_agents, 'Incorrect number of agents found in initial world'

    # Get value of block
    def getBlock(self, coord):
        # change coordinates to int
        coord = np.array(coord, dtype=int)

        if (coord < 0).any() or (coord >= self.shape).any():
            return -3
        return self.state[coord[0], coord[1], coord[2]]

    # Set block to input value
    def setBlock(self, coord, val):
        # change coordinates to int
        coord = np.array(coord, dtype=int)

        if (coord < 0).any() or (coord >= self.shape).any():
            return False
        self.state[coord[0], coord[1], coord[2]] = val
        return True

    # Swap two blocks
    def swap(self, coord1, coord2, agent_id):
        temp = self.getBlock(coord1)
        if temp == -2:
            self.setBlock(coord2, -1)
        else:
            if self.getBlock(coord2) == -2:
                self.setBlock(coord1, 0)
            else:
                self.setBlock(coord1, self.getBlock(coord2))
                self.setBlock(coord2, temp)

    # Get value of block
    def getPos(self, agent_id):
        # change coordinates to int
        coord = np.array(self.agents_pos[agent_id], dtype=int)

        return coord

    def setPos(self, new_pos, agent_id):
        self.agents_pos[agent_id] = new_pos
        npx, npy, npz = int(new_pos[0]), int(new_pos[1]), int(new_pos[2])
        assert self.state[npx, npy, npz] == agent_id, "Problem: agent {}'s position in agents_pos does not seem to match world.state ({})".format(agent_id, self.getBlock(new_pos))

    # Return predicted new state after action (Does not actually execute action, may be an invalid action)
    def act(self, action, agent_id):
        current_state = self.getPos(agent_id)
        new_state = current_state.copy()

        # Move
        if action in range(1,5):
            new_state[0:3] += self.heading2vec(action-1)

        return new_state

    # Get observation
    def getObservation(self, coord, ob_range):
        '''
        Observation: Box centered around agent position
            (returns -3 for blocks outside world boundaries)

        args:
            coord: Position of agent. Numpy array of length 3.
            ob_range: Vision range. Numpy array of length 3.

        note: observation.shape is (2*ob_range[0]+1, 2*ob_range[1]+1, 2*ob_range[2]+1)
        '''
        if (ob_range == [-1, -1, -1]).all(): # see EVERYTHING
            world_state = self.state
        else:
            ob = -3*np.ones([2*ob_range[0]+1, 2*ob_range[1]+1, 2*ob_range[2]+1])

            # change coordinates to int
            coord = np.array(coord, dtype=int)

            # two corners of view in world coordinate
            c0 = coord - ob_range
            c1 = coord + ob_range

            # clip according to world boundaries
            c0_c = np.clip(c0, [0,0,0], self.shape)
            c1_c = np.clip(c1, [0,0,0], self.shape)

            # two corners of view in observation coordinates
            ob_c0 = c0_c - coord + ob_range
            ob_c1 = c1_c - coord + ob_range

            # assign data from world to observation
            world_state = self.state[c0_c[0]:c1_c[0]+1, c0_c[1]:c1_c[1]+1, c0_c[2]:c1_c[2]+1]
            ob[ob_c0[0]:ob_c1[0]+1, ob_c0[1]:ob_c1[1]+1, ob_c0[2]:ob_c1[2]+1] = world_state

        return world_state

    # Compare with a plan to determine job completion
    def done(self, state_obj):
        blocks_state = np.asarray(np.clip(self.state, -1., 0.), dtype=int)
        blocks_plan  = np.asarray(np.clip( state_obj, -1., 0.), dtype=int)

        is_built = np.sum(blocks_state * blocks_plan) == -np.sum(blocks_plan) # All correct blocks are placed
        done = (blocks_state == blocks_plan).all()

        return done, is_built
    
    def countExtraBlocks(self, state_obj):
        blocks_state = np.asarray(np.clip(self.state, -1., 0.), dtype=int)
        blocks_plan  = np.asarray(np.clip( state_obj, -1., 0.), dtype=int)

        return (np.sum(blocks_plan) - np.sum(blocks_state))

    # Transform heading to x, z
    def heading2vec(self, fac):
        dx = ((fac + 1) % 2)*(1 - fac)
        dy = 0
        dz = (fac % 2)*(2 - fac)
        return np.asarray([dx,dy,dz])



class MinecraftEnv(gym.Env):
    '''
    3D Grid Environment
        Observation: (OrderedDict)
            Position:   X, Y, Z  (+Y = up)
            Action heading:     {0:+Z, 1:+X, 2:-Z, 3:-X}
            View:       A box centered around the agent (limited view)
                block = -1
                air = 0
                agent = 1
                out of world range = -3
        Action space: (Tuple)
            agent_id: positive integer (always 1)
            action: {0:NOP, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST, 5:PICK_NORTH,
            6:PICK_EAST, 7:PICK_SOUTH, 8:PICK_WEST, 9:PLACE_NORTH, 10:PLACE_EAST, 11:PLACE_SOUTH, 12:PLACE_WEST}
            NORTH: +Y, EAST: +X
        Reward: -0.1 for each action, +5 for each block correctly placed
    '''
    metadata = {"render.modes": ["human", "ansi"]}

    # Initialize env
    def __init__(self, num_agents=1, observation_range=1, observation_mode='id_visible', world0=None, FULL_HELP = False, MAP_ID=1):
        """
        Args:
            observation_range: Integer for cube. List of length 3 for box.
            observation_mode: {'default', 'id_visible'}
        """
        # Parse input parameters and check if valid
        #   observation_range
        if type(observation_range) is int:
            ob_range = observation_range*np.ones(3, dtype=int)
        else:
            assert len(observation_range) == 3, 'Wrong number of dimensions for \'observation_range\''
            ob_range = np.array(observation_range)

        #   observation_mode
        assert observation_mode in ['default', 'id_visible'], 'Invalid \'observation_mode\''

        # Initialize member variables
        self.num_agents = num_agents
        #self.ob_shape = 2*ob_range + 1
        self.ob_range = ob_range
        self.ob_mode = observation_mode
        self.finished = False
        self.mutex = Lock()
        self.fresh = False

        self.FULL_HELP = FULL_HELP # Defines if we help agent identify its next goal
        self.map_id = MAP_ID-1
        self.RANDOMIZED_PLANS = (MAP_ID == 0) # Defines if we randomize the plans during training
        
        # Initialize data structures
        self.world_shape = (10,4,10)
        self._setObjective()
        if world0 is None:
            self._setInitial()
        else:
            self.state_init = world0

        # Check everything is alright
        assert self.state_init.shape == self.state_obj.shape, '\'state_init\' and \'state_obj\' dimensions do not match'
        self.world = Grid3DState(self.state_init, self.num_agents)

        self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(13)])
        self.viewer = None

    # Define objective world here
    def _setObjective(self):
        '''
        Objective state of the world (3d numpy array)
            air = 0
            block = -1
        '''
        plan_map = np.zeros(self.world_shape)

        if self.RANDOMIZED_PLANS:
            p_plan = np.random.uniform(0.05, 0.5)

            while np.sum(plan_map[:,0,:]) == 0:
                randPlan = - np.random.binomial(1, p_plan, size=self.world_shape)
                plan_map[:,0,:] = randPlan[:,0,:]

            # source block (nothing around to allow pickup)
            for pos in SOURCES:
                plan_map[pos[0], pos[1], pos[2]] = -2
                if pos[0]-1 >= 0:
                    plan_map[pos[0]-1, pos[1], pos[2]] = 0
                if pos[0]+1 < self.world_shape[0]:
                    plan_map[pos[0]+1, pos[1], pos[2]] = 0
                if pos[2]-1 >= 0:
                    plan_map[pos[0], pos[1], pos[2]-1] = 0
                if pos[2]+1 < self.world_shape[2]:
                    plan_map[pos[0], pos[1], pos[2]+1] = 0

            # Other random blocks
            for j in range(1, self.world_shape[1]-1): # blocks cannot be placed at the highest level
                # Let's place blocks on level j on top of blocks on level j-1 only
                plan_map[:,j,:] = (plan_map[:,j-1,:] == -1).astype(int) * randPlan[:,j,:]
        else:
            # Place blocks on world plan
            for pos in PLAN_MAPS[self.map_id]:
                plan_map[pos[0], pos[1], pos[2]] = -1

            # source block (nothing around to allow pickup)
            for pos in SOURCES:
                plan_map[pos[0], pos[1], pos[2]] = -2
                if pos[0]-1 >= 0:
                    plan_map[pos[0]-1, pos[1], pos[2]] = 0
                if pos[0]+1 < self.world_shape[0]:
                    plan_map[pos[0]+1, pos[1], pos[2]] = 0
                if pos[2]-1 >= 0:
                    plan_map[pos[0], pos[1], pos[2]-1] = 0
                if pos[2]+1 < self.world_shape[2]:
                    plan_map[pos[0], pos[1], pos[2]+1] = 0

        self.state_obj = plan_map

    # Define initial agent distribution here
    def _setInitial(self, empty=False, full=False):
        '''
        Initial state of the world (3d numpy array)
            air = 0
            block = -1
            source = -2
            agent = agent_id (always 1)
        '''
        # Randomized world based on self.state_obj
        #p_sparse, p_plan = 0.1, 0.4
        if full:
            p_sparse, p_plan = np.random.uniform(0., 0.5), 1.
        else:
            p_sparse, p_plan = np.random.uniform(0., 0.3), np.random.uniform(0., 1.)
        randSparse = np.random.binomial(1, p_sparse, size=self.world_shape)
        randPlan = np.random.binomial(1, p_plan, size=self.world_shape)

        world = np.zeros(self.world_shape)
        if not empty:
            world[:,0,:] = self.state_obj[:,0,:] * randPlan[:,0,:] + (-1-self.state_obj[:,0,:]) * randSparse[:,0,:]

        # source block (nothing around to allow pickup)
        for pos in SOURCES:
            world[pos[0], pos[1], pos[2]] = -2
            if pos[0]-1 >= 0:
                world[pos[0]-1, pos[1], pos[2]] = 0
            if pos[0]+1 < self.world_shape[0]:
                world[pos[0]+1, pos[1], pos[2]] = 0
            if pos[2]-1 >= 0:
                world[pos[0], pos[1], pos[2]-1] = 0
            if pos[2]+1 < self.world_shape[2]:
                world[pos[0], pos[1], pos[2]+1] = 0

        # agents: Random initial position
        for i in range(self.num_agents):
            rx, ry, rz = np.random.randint(self.world_shape[0]), np.random.randint(2), np.random.randint(self.world_shape[2])
            while not (world[rx,ry,rz] == 0 and ((ry == 0) or (ry > 0 and world[rx,ry-1,rz] == -1))):
                rx, ry, rz = np.random.randint(self.world_shape[0]), np.random.randint(self.world_shape[1]), np.random.randint(self.world_shape[2])
            world[rx,ry,rz] = i+1

        if not empty:
            # Other random blocks
            for j in range(1, self.world_shape[1]-1): # blocks cannot be placed at the highest level
                # Where are agents on level j
                agentMap = (world[:,j,:] > 0).astype(int) * world[:,j,:]
                # We can place blocks either on agents, or on blocks that are not themselves on agents. Also, let's not place blocks were agents are...
                if j < 2:
                    prevMap = (1-np.clip(agentMap,0,1)) * np.clip((world[:,j-1,:] > 0).astype(int) + (world[:,j-1,:] == -1).astype(int), 0, 1)
                else:
                    prevMap = (1-np.clip(agentMap,0,1)) * np.clip((world[:,j-1,:] > 0).astype(int) + (world[:,j-1,:] == -1).astype(int) * (world[:,j-2,:] == -1).astype(int), 0, 1)
                # Let's place blocks on level j
                world[:,j,:] = self.state_obj[:,j,:] * prevMap * randPlan[:,j,:] + (-1-self.state_obj[:,j,:]) * prevMap * randSparse[:,j,:] + agentMap

        self.state_init = world

    # Returns an observation of an agent
    def _observe(self, agent_id):
        # Get agent states
        agent_pos = self.world.getPos(agent_id)

        # Get world observation
        ob_view = self.world.getObservation(agent_pos, self.ob_range)
        if self.ob_mode == 'default':
            ob_view = np.clip(ob_view, -3, 1)

        # 1. Position map (one-hot matrix, gives agent's position)
        pos_map = np.zeros(self.world_shape)
        px, py, pz = int(agent_pos[0]), int(agent_pos[1]), int(agent_pos[2])
        pos_map[px,py,pz] = 1

        # 2. All agents map (air and anonymous agents info only)
        agents_map = np.clip(ob_view, 0, 1)

        # 3. Block map (blocks only)
        blocks_map = np.clip(ob_view, -1, 0)
        for pos in SOURCES: # Remove sources from block map
            blocks_map[pos[0], pos[1], pos[2]] = 0

        # 4. Sources map (sources only)
        sources_map = np.zeros(self.world_shape)
        for pos in SOURCES:
            sources_map[pos[0], pos[1], pos[2]] = -2

        # 5. Global plan map
        plan_map = self.state_obj.copy()
        for pos in SOURCES:
            plan_map[pos[0], pos[1], pos[2]] = 0

        return [pos_map, agents_map, blocks_map, sources_map, plan_map]

    # Resets environment
    def _reset(self, agent_id, empty=False, full=False):
        self.finished = False
        self.mutex.acquire()

        if not self.fresh:
            # Check everything is alright
            assert self.state_init.shape == self.state_obj.shape, '\'state_init\' and \'state_obj\' dimensions do not match'

            # Initialize data structures
            self._setObjective()
            self._setInitial(empty=empty, full=full)
            self.world = Grid3DState(self.state_init, self.num_agents)
            #self._initSpaces()

            self.fresh = True
            self.finalAgentID = 0

        _, is_built = self.world.done(self.state_obj)
        has_block = self.world.getBlock(self.world.getPos(agent_id) + np.array([0,1,0])) == -1
        
        self.mutex.release()
        
        return self._listNextValidActions(agent_id), has_block, is_built

    # Executes an action by an agent
    def _step(self, action_input):
        self.fresh = False

        # Check action input
        assert len(action_input) == 2, 'Action input should be a tuple with the form (agent_id, action)'
        assert action_input[1] in range(13), 'Invalid action'
        assert action_input[0] in range(1, self.num_agents+1)

        # Parse action input
        agent_id = action_input[0]
        action   = action_input[1]

        # Lock mutex (race conditions start here)
        self.mutex.acquire()
        initDone = self.finished

        # Get current agent state
        agent_pos = self.world.getPos(agent_id)

        # Get estimated new agent state
        new_agent_pos = self.world.act(action, agent_id)

        # Execute action & determine reward
        reward = ACTION_COST

        if action in range(1,5):     # Move
            validAction = False # Valid Movement ?

            # get coordinates and blocks near new position
            new_pos = new_agent_pos
            new_pos_upper = new_pos + np.array([0,1,0])
            new_pos_lower = new_pos + np.array([0,-1,0])
            new_pos_lower2 = new_pos + np.array([0,-2,0])

            block_newpos = self.world.getBlock(new_pos)
            block_upper = self.world.getBlock(new_pos_upper)
            block_lower = self.world.getBlock(new_pos_lower)
            block_lower2 = self.world.getBlock(new_pos_lower2)

            # execute movement if valid
            if block_newpos == 0:  # air in front?
                if block_lower == -1 or block_lower == -3:    # block or ground beneath?
                    dest = np.array(new_pos, dtype=int)
                    if (self.world.state[dest[0], 0:dest[1], dest[2]] == -1).all():
                        new_agent_pos = new_pos    # horizontal movement
                        validAction = True
                elif block_lower == 0 and block_lower2 in [-1, -3]:   # block or ground beneath?
                    dest = np.array(new_pos_lower, dtype=int)
                    if (self.world.state[dest[0], 0:dest[1], dest[2]] == -1).all():
                        new_agent_pos = new_pos_lower  # downstairs movement
                        validAction = True
            elif block_newpos == -1 and block_upper == 0:   # block in front and air above?
                dest = np.array(new_pos_upper, dtype=int)
                if (self.world.state[dest[0], 0:dest[1], dest[2]] == -1).all():
                    new_agent_pos = new_pos_upper    #upstairs movement
                    validAction = True

            # Prevent agents from accessing the highest level
            if new_agent_pos[1] == self.world_shape[1]-1:
                validAction = False

            if validAction:
                self.world.swap(agent_pos, new_agent_pos, agent_id)
                self.world.swap(agent_pos + np.array([0,1,0]), new_agent_pos + np.array([0,1,0]), agent_id)
                self.world.setPos(new_agent_pos, agent_id)

        elif action in range(5,13):       # Pick & Place
            # determine block movement
            top = agent_pos + np.array([0,1,0])
            front = agent_pos + self.world.heading2vec((action-1) % 4)

            if action < 9: # pick
                source = front
                dest = top
            else:
                source = top
                dest = front
            above_source = source + np.array([0,1,0])
            dest = np.array(dest, dtype=int)
            source = np.array(source, dtype=int)

            # execute
            if self.world.getBlock(source) in [-1, -2] and self.world.getBlock(dest) in [0, -2] and self.world.getBlock(above_source) in [0, -3] and (action < 9 or (action > 8 and (self.world.state[dest[0], 0:dest[1], dest[2]] == -1).all())) and not (action < 9 and source[1] == self.world_shape[1]-1):
                if self.world.getBlock(dest) == -2: # Blocks can be destroyed by placing them in a source. However, we
                    self.world.setBlock(source, 0)  # should not use swap in this case (or agents will pick up the source)

                    if self.FULL_HELP:
                        if np.sum(np.clip(self.world.state, -1, 0)) > np.sum(np.clip(self.state_obj, -1, 0)):
                            reward -= PLACE_REWARD
                        elif np.sum(np.clip(self.world.state, -1, 0)) < np.sum(np.clip(self.state_obj, -1, 0)):
                            reward += PLACE_REWARD
                elif self.world.getBlock(source) == -2: # Make a block appear above the agent
                    self.world.setBlock(dest, -1)

                    if self.FULL_HELP:
                        if np.sum(np.clip(self.world.state, -1, 0)) < np.sum(np.clip(self.state_obj, -1, 0)):
                            reward -= PLACE_REWARD
                        elif np.sum(np.clip(self.world.state, -1, 0)) > np.sum(np.clip(self.state_obj, -1, 0)):
                            reward += PLACE_REWARD
                else:
                    self.world.swap(source, dest, agent_id)

                    # place/pick incorrect block creates additional +/- rewards only once plan is completed, to encourage cleanup
                    _, complete = self.world.done(self.state_obj)

                    if action > 8 and self.state_obj[dest[0], dest[1], dest[2]] == -1: # Place correct block
                        reward += PLACE_REWARD * (dest[1]+1)**2
                    elif action < 9 and self.state_obj[source[0], source[1], source[2]] == -1: # Removing correct block
                        reward -= PLACE_REWARD * (source[1]+1)**2
                    elif action > 8 and self.state_obj[dest[0], dest[1], dest[2]] == 0 and complete: # Place incorrect block
                        reward -= PLACE_REWARD
                    elif action < 9 and self.state_obj[source[0], source[1], source[2]] == 0 and complete: # Remove incorrect block
                        reward += PLACE_REWARD

        # Perform observation
        state = self._observe(agent_id) # ORIGINAL 5-TENSOR STATE

        # Done?
        done, is_built = self.world.done(self.state_obj)
        self.finished |= done
        if initDone != self.finished:
            assert(self.finalAgentID == 0)
            self.finalAgentID = agent_id

        # Additional info
        info = self._listNextValidActions(agent_id, action)
        has_block = self.world.getBlock(self.world.getPos(agent_id) + np.array([0,1,0])) == -1

        # Unlock mutex
        self.mutex.release()

        return state, reward, done, info, has_block, is_built

    def _getReward(self, reward_factor = 0.02):
        # Calculate number of correct/incorrect blocks
        good_blocks = 0
        for pos in PLAN_MAPS[self.map_id]:
            good_blocks += int(self.world.state[pos[0], pos[1], pos[2]] == -1) * (pos[1]+1)**2 # Squaring encourages the creation of ramps

        extra_blocks = (1 + np.clip(self.state_obj, -1, 0)) * np.clip(np.array(self.world.state), -1, 0) # Clip removes agent and sources
        bad_blocks = abs(np.sum(extra_blocks))

        assert good_blocks >= 0
        assert bad_blocks >= 0
        return (good_blocks - reward_factor * bad_blocks)

    def _listNextValidActions(self, agent_id, prev_action=0):
        available_actions = [] # NOP always allowed

        # Get current agent state
        agent_pos = self.world.getPos(agent_id)

        for action in range(1,5):     # Move
            validAction = False

            # Get estimated new agent state
            new_agent_pos = self.world.act(action, agent_id)

            # get coordinates and blocks near new position
            new_pos = new_agent_pos
            new_pos_upper = new_pos + np.array([0,1,0])
            new_pos_lower = new_pos + np.array([0,-1,0])
            new_pos_lower2 = new_pos + np.array([0,-2,0])

            block_newpos = self.world.getBlock(new_pos)
            block_upper = self.world.getBlock(new_pos_upper)
            block_lower = self.world.getBlock(new_pos_lower)
            block_lower2 = self.world.getBlock(new_pos_lower2)

            # execute movement if valid
            if block_newpos == 0:  # air in front?
                if block_lower == -1 or block_lower == -3:    # block or ground beneath?
                    dest = np.array(new_pos, dtype=int)
                    if (self.world.state[dest[0], 0:dest[1], dest[2]] == -1).all():
                        new_agent_pos = new_pos    # horizontal movement
                        validAction = True
                elif block_lower == 0 and block_lower2 in [-1, -3]:   # block or ground beneath?
                    dest = np.array(new_pos_lower, dtype=int)
                    if (self.world.state[dest[0], 0:dest[1], dest[2]] == -1).all():
                        new_agent_pos = new_pos_lower  # downstairs movement
                        validAction = True
            elif block_newpos == -1 and block_upper == 0:   # block in front and air above?
                dest = np.array(new_pos_upper, dtype=int)
                if (self.world.state[dest[0], 0:dest[1], dest[2]] == -1).all():
                    new_agent_pos = new_pos_upper    #upstairs movement
                    validAction = True

            # Prevent agents from accessing the highest level
            if new_agent_pos[1] == self.world_shape[1]-1:
                validAction = False

            if validAction:
                available_actions.append(action)

        for action in range(5,13):       # Pick & Place
            # determine block movement
            top = agent_pos + np.array([0,1,0])
            front = agent_pos + self.world.heading2vec((action-1) % 4)

            if action < 9:
                source = front
                dest = top
            else:
                source = top
                dest = front
            above_source = source + np.array([0,1,0])
            dest = np.array(dest, dtype=int)
            source = np.array(source, dtype=int)

            # execute
            if self.world.getBlock(source) in [-1, -2] and self.world.getBlock(dest) in [0, -2] and self.world.getBlock(above_source) in [0, -3] and (action < 9 or (action > 8 and (self.world.state[dest[0], 0:dest[1], dest[2]] == -1).all())) and not (action < 9 and source[1] == self.world_shape[1]-1):
                available_actions.append(action)

        if len(available_actions) > 1 and opposite_actions[prev_action] in available_actions:
            available_actions.remove(opposite_actions[prev_action])
        elif len(available_actions) == 0: # Only allow NOP if nothing else is valid
            available_actions.append(0)

        return available_actions
    
    # Render gridworld state
    def _render(self, agent_id=1, mode='human', close=False):
        world = self.world  # world = self.world.getObservation(agent_pos, self.ob_range)
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        depth = self.world.shape[1]
        min_size = 10 # minimum radius of the smallest square
        screen_width = 500
        screen_height = 500
        square_width = screen_width / world.shape[0]
        square_height = screen_height / world.shape[2]
        min_size = min(min_size, min(square_width, square_height))
        square_width_offset = (square_width-min_size) / depth
        square_height_offset = (square_height-min_size) / depth

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width,screen_height)
            self.squares = [[[rendering.FilledPolygon([(i*square_width+square_width_offset*k,(j+1)*square_height-square_height_offset*k),
                                                      (i*square_width+square_width_offset*k,j*square_height+square_height_offset*k),
                                                      ((i+1)*square_width-square_width_offset*k, j*square_height+square_height_offset*k),
                                                      ((i+1)*square_width-square_height_offset*k, (j+1)*square_height-square_height_offset*k)])
                              for k in range(self.world.shape[1])]
                            for i in range(self.world.shape[0])]
                           for j in range(self.world.shape[2])]
            for row in self.squares:
                for square in row:
                    for subsquare in square:
                        self.viewer.add_geom(subsquare)


        if self.world.state is None: return None

        for x in range(world.shape[0]):
            for y in range(world.shape[2]):
                for z in reversed(range(world.shape[1])):
                    val = world.getBlock([x, z, y])
                    new_color = AIR
                    if val == -2: # block spawn
                        new_color = BLOCK_SPAWN
                    elif val == -1:
                        new_color = BLOCK
                    elif val > 0:
                        new_color = AGENT
                    elif val != 0:
                        print('Error in map at {},{},{}, val = {}'.format(x,z,y,val))

                    if self.state_obj[x,z,y] == -1:
                        new_color = (new_color*4+PLAN_COLOR) / 5
                    if val == 0 and z != 0:
                        if self.state_obj[x,z,y] == -1:
                            self.squares[y][x][z]._color.vec4 = (new_color[0], new_color[1], new_color[2], 0.5)
                        else:
                            self.squares[y][x][z]._color.vec4 = (0,0,0,0)
                    else:
                        self.squares[y][x][z].set_color(*(new_color))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
