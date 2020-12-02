"""
Basic data structures for modeling information propagation.
"""

import warnings
import numpy as np
import scipy.sparse
import networkx as nx
import matplotlib.pyplot as plt
import itertools


class Agent:
    
    all_agents = list()

    @classmethod
    def to_dict(cls, agents):
        """
        Helper method to coerce an input to a dictionary keyed by agent_id.
        """
        if agents is None:
            return dict()
        elif isinstance(agents, list):
            return {agent.id:agent for agent in agents}
        elif isinstance(agents, dict):
            return agents
        elif isinstance(agents, cls):
            return {agents.id:agents}
        else:
            raise ValueError("Agents.to_dict(agents) expects list, dict, single Agent, or None.")

    @classmethod
    def to_list(cls, agents):
        """
        Helper method to coerce an input to a list of (unique) agents.
        """
        if agents is None:
            return list()
        elif isinstance(agents, list):
            return list(set(agents))
        elif isinstance(agents, dict):
            return list(set(agents.values()))
        elif isinstance(agents, cls):
            return [agents]
        else:
            raise ValueError("Agents.to_list(agents) expects list, dict, single Agent, or None.")


    def __init__(self,
        env = None,
        workplace_ids = None,
        specialty_ids = None,
        inner_circle = None,
        outer_circle = None,
        informed_init = False,
        receptivity = 0.5,
        persuasiveness = 0.5,
    ):
        """

        Parameters
        ----------
            env:
                The simulation environment this agent is bound to.

            informed_init:
                A boolean indivating whether this individual begins informed or not.

            workplace_ids:
                A list of id's of the workplaces this indiviudal belongs to.

            specialty_ids:
                A list of the specialties this individual is part of.

            inner_circle:
                List of id's of agents in this professional's close network.

            outer_circle:
                List of id's of agents in this professional's extended network.

            receptivity:
                A float between 0 and 1 indicating this professional's propensity to receive information.

            persuasiveness:
                A float between 0 and 1 indicating this individual's propensity to convey information.

        """

        # Generate unique ID and add to list:
        self.id = len(Agent.all_agents)
        Agent.all_agents.append(self)

        # Define properties:
        self.env = env
        self.informed_init = informed_init
        self.workplace_ids = workplace_ids if workplace_ids else list()
        self.specialty_ids = specialty_ids if specialty_ids else list()
        self.inner_circle = inner_circle if inner_circle else list()
        self.outer_circle = outer_circle if outer_circle else list()
        self.receptivity = max(0,min(1,receptivity)) * 1.0
        self.persuasiveness = max(0,min(1,persuasiveness)) * 1.0

        # Define state variables:
        self.informed = self.informed_init  # Is this professional currently up to date?
        self.intervention = False  # Has this profession received a direct intervention?


class AdjacencyMatrix:
    """
    A representation of the ajacency matrix between agents, stored as a sparse boolean matrix,
    indicating whether agent_i has a (not necessarily symmetric) relationship to agent_j.
    This data structure is most efficient when initialized with all agents, but alternatively
    the user can use the `add_connection` and `update` methods to manually add connections.
    Self-connections (i.e. diagonal entries) should always be False.
    """

    def __init__(self, env, scope='inner', agent_ids=None):
        """
        env:
            The simulation environment.
        scope:
            Which type of relationship circle to encode
            (i.e. 'inner', 'outer', or 'both').
        agent_ids:
            (Optional) List of agent_ids in the order they should be stored in the matrix.
            If not provided, defaults to the order used by the environment.
        """

        # Bind simulation environment:
        self.env = env

        # Set whether or not this matrix encodes inner or outer circle of relationships:
        self.valid_scopes = {'inner','outer','both'}
        self.scope = scope
        assert scope in self.valid_scopes, f"{scope} is not a valid scope: {self.valid_scopes}"

        # Maintain a list of agent_ids in the order the appear in the matrix:
        self.agent_ids = agent_ids if agent_ids else self.env.agent_ids

        # Keep track of matrix in scipy.sparse.csr_matrix (or None when it needs rebuilding):
        self.matrix = None

        # Initialize:
        self.update()

    def add_connection(self, i, j):
        # Check i:
        try:
            i = i.id
        except:
            assert isinstance(i, int), "`i` should be an Agent object or an agent_id"
        assert i in self.env.agents.keys(), f"{i} is not one of the Agents in this environment."
        # Check j:
        try:
            j = j.id
        except:
            assert isinstance(j, int), "`j` should be an Agent object or an agent_id"
        assert j in self.env.agents.keys(), f"{j} is not one of the Agents in this environment."
        # Update agent:
        if self.scope in {'inner','both'}:
            self.env.agents[i].inner_circle.append(j)
        if self.scope in {'outer','both'}:
            self.env.agents[i].outer_circle.append(j)

    def update(self):
        
        # Build sparse matrix (build as lil_matrix matrix and convert to csr_matrix):
        n_agents = len(self.agent_ids)
        self.matrix = scipy.sparse.lil_matrix((n_agents,n_agents), dtype=bool)

        # Populate matrix:
        for i,i_agent_id in enumerate(self.agent_ids):
            i_agent = self.env.agents[i_agent_id]
            # Get list of agents this agent is connected to:
            j_agent_ids = list()
            if self.scope in {'inner','both'}:
                j_agent_ids.extend(i_agent.inner_circle)
            if self.scope in {'outer','both'}:
                j_agent_ids.extend(i_agent.outer_circle)
            j_agent_ids = list(set(j_agent_ids))
            for j_agent_id in j_agent_ids:
                j = self.agent_ids.index(j_agent_id)
                self.matrix[i,j] = True
        # Convert to CSR encoding:
        self.matrix = self.matrix.tocsr()

        return self.matrix

    def toarray(self):
        return self.matrix.toarray()

    @property
    def n_connections(self):
        # Return size of matrix (only True entries are stored):
        return self.matrix.size


class InfluenceMatrix:
    """
    A representation of each agent_i's probability of influencing agent_j,
    conditional on agent_i having the information before the encounter
    and regardless of whether or not agent_j is already informed.
    The values are stored as floats (between 0 and 1) in a sparse matrix,
    and capture a combination of the characteristics of the professional network
    (i.e. how likely each agent is to encounter each other agent) and the
    characteristics of each agent (i.e. how likely the agents are to
    spread/receive information).
    The influence matrix is determined by the networks structure and
    environment assumptions/hyperparameters, and should remain
    agnostic to the state/action history.
    """

    def __init__(self, env, model=None, agent_ids=None):
        """
        env:
            The simulation environment.
        method:
            Hyperparameter to control which model is used.
        agent_ids:
            (Optional) List of agent_ids in the order they should be stored in the matrix.
            If not provided, defaults to the order used by the environment.
        """
        self.env = env
        # Set influence model:
        self.valid_models = {'default'}
        self.model = model if model is not None else 'default'
        assert self.model in self.valid_models, f"{model} is not a valid model: {self.valid_models}"

        # Maintain a list of agent_ids in the order the appear in the matrix:
        self.agent_ids = agent_ids if agent_ids else self.env.agent_ids

        # Keep track of matrix in scipy.sparse.csr_matrix (or None when it needs rebuilding):
        self.matrix = None

        # Initialize:
        self.update()

    def update(self):

        if self.model=='default':

            # Start from adjacency matrices:
            inner = self.env.inner.matrix  #.copy()
            outer = self.env.outer.matrix  #.copy()

            # Remove inner circle from outer cirlce (i.e. remove redundancy):
            outer = ((outer*1.0 - inner*1.0)>0)

            # Get agent properies (in same order as adjacency matrix):
            agents = [self.env.agents[agent_id] for agent_id in self.env.agent_ids]
            receptivity = np.array([agent.receptivity for agent in agents]).reshape(1,-1)  # Row vector.
            persuasiveness = np.array([agent.persuasiveness for agent in agents]).reshape(-1,1)  # Column vector.

            # Scale by receptivity (apply across all rows, with different value for each column):
            inner = inner.multiply(receptivity)
            outer = outer.multiply(receptivity)

            # Scale by persuasiveness (apply across all columns, with different value for each row):
            inner = inner.multiply(persuasiveness)
            outer = outer.multiply(persuasiveness)

            # Combine inner and outer circles:
            # (no values will be more than 1 if inner and outer were never True for the same entry)
            matrix = inner + 0.5*outer

            # Store result:
            self.matrix = matrix

        else:
            raise NotImplementedError(f"Influence model {self.model} is not yet implemented.")

        return self.matrix
    
    def toarray(self):
        return self.matrix.toarray()


class TransitionMatrix:
    """
    A representation of the transition probabilities where:
    - the state space is each possible combination status values across agents and
    - the action space is each possible combination of agents selected for intervention.
    Agents can become informed organically (from adjacent agents in the network)
    or by being selected for an intervention. We assume the intervention stage happens
    after then organic propagation (i.e. an agent selected for intervention
    does not begin to influence neighbors until the next stage, unless they happened to
    already be informed at the start of the stage, i.e. if the intervention was superfluous).
    """

    @classmethod
    def enumerate_actions(cls, env, n_selected=None, state=None, as_objects=False):
        """
        Enumerate all possible inteventions in a given environment.
        env:
            The simulation environment.
        n_selected:
            (int) The max number of inteventions (defaults to value set in environment).
        state:
            (State object) If specified, enumerates only actions that can be meaningfully taken at this state.
            If False, exhaustively list all possible actions (including those that select already informed agents).
        as_objects:
            (bool) Optionally return a list of Action objects.
            Otherwise, return a list of boolean arrays (where values are in the same order as env.agent_ids).
        """
        # Get intervention size:
        n_selected = n_selected if n_selected else env.intervention_size
        n_selected = int(max(0,min(n_selected, len(env.agent_ids))))
        if state is None:
            # Consider all agents as candidates for intervention:
            selections = list(itertools.combinations(env.agent_ids, n_selected))
        elif state is not None:
            # Convert to State object (accepts various input formats):
            state = State.coerce(env=env, state=state)
            # Get agent_ids of uninformed agents:
            candidates = np.array(env.agent_ids)[np.argwhere(state.vector==False)].flatten()
            # Assume more inteventions are better than fewer:
            n_selected = min(n_selected, len(candidates))
            # Get all possible combinations:
            selections = itertools.combinations(candidates, n_selected)
        # Build an action object (which expects an list of agent_ids):
        actions = [Action(env, selected_ids=list(selection)) for selection in selections]
        if not as_objects:
            # Optionally extract the internal representation (vector of booleans):
            actions = [action.vector for action in actions]
        return actions

    @classmethod
    def enumerate_states(cls, env, state=None, as_objects=False):
        """
        Enumerate the possible states for a given environment.
        env:
            The simulation environment.
        state:
            (State object) If specified, enumerates only states that can be reached from this state.
            If False, exhaustively list all possible states.
        as_objects:
            (bool) Optionally return a list of State objects.
            Otherwise, return a list of boolean arrays (where values are in the same order as env.agent_ids).
        """
        if state is None:
            # Excaustive case:
            states = list(itertools.product([False,True], repeat=len(env.agent_ids)))
            states = [np.array(state) for state in states]
        elif state is not None:
            # Pruned case:
            # Convert to State object (accepts various input formats):
            current_state = State.coerce(env=env, state=state)
            # Enumerate states (as list of tuples) and convert to arrays:
            states = list(itertools.product([False,True], repeat=len(env.agent_ids)))
            states = [np.array(state) for state in states]
            # Remove unreachable states:
            # A state is unreachable there is any agent who is informed in the current state
            # would become uninformed in the candidate state.
            states = [
                candidate_state for candidate_state in states
                if not np.any( (current_state==True)&(candidate_state==False) )
            ]
        if as_objects:
            # Build an action object (using a dict of booleans):
            states = [
                State(env=env, vector=state)
                for state in states
            ]
        return states

    @classmethod
    def agent_probabilities(cls, env, state, action):
        """
        Given a starting state and an action, returns the probability that each agent is informed (in the landing state).
        The return is a vector of floats, corresponding to the agents in env.agent_id order.
        env:
            The simulation environment.
        state:
            The starting state.
        action:
            The applied action.
        """

        # Get boolean representation of state:
        state = State.coerce(env=env, state=state).vector  # Vector of booleans.
        # Get boolean representation of action:
        action = Action.coerce(env=env, action=action).vector  # Vector of booleans.

        # Get state vector and influence matrix:
        probs = env.influence.matrix

        # Convert to numpy matrix (won't remain sparse during manipulation):
        probs = probs.toarray()

        # Multiply by state column -- uninformed agents will not influence anyone:
        probs = np.where(state.reshape(-1,1), probs, 0)

        # Multiply by state row -- already informed agents will remain informed:
        probs = np.where(state.reshape(1,-1), 1, probs)

        # Multiply by action row -- agents selected for intervention will be informed:
        probs = np.where(action.reshape(1,-1), 1, probs)

        # Calculate how likely each agent is to be informed at the end of this step:
        probs = 1-np.prod(1-probs,axis=0)

        return probs

    def __init__(self, env, model=None, n_selected=None, agent_ids=None):
        """
        env:
            The simulation environment.
        method:
            Hyperparameter to control which model is used.
            The 'exhaustive' mode enumerates all states and actions (using loops).
            The 'exhaustive_fast' mode enumerates all states and actions (using matrix operations).
            The 'reachable' mode limits enumeration to meaningful actions, from reachable states to reachable states.
            The 'pruned' mode limits enumeration to meaningful actions, from current state to reachable states.
        n_selected:
            The (max) number of agents to select for each intervention.
        agent_ids:
            (Optional) List of agent_ids in the order they should be stored in the state/acton space.
            If not provided, defaults to the order used by the environmnet.
        """
        self.env = env
        # Set influence model:
        self.valid_models = {'exhaustive','exhaustive_fast','reachable','pruned'}
        self.model = model if model is not None else 'exhaustive_fast'
        assert self.model in self.valid_models, f"{model} is not a valid transition model: {self.valid_models}"

        # Maintain a list of agent_ids in the order the appear in the matrix:
        self.agent_ids = agent_ids if (agent_ids is not None) else list(sorted(env.agents.keys()))

        # Get intervention size:
        n_selected = n_selected if n_selected else env.intervention_size
        n_selected = int(max(0,min(n_selected, len(env.agent_ids))))
        self.n_selected = n_selected

        # Keep track of matrix in scipy.sparse.csr_matrix (or None when it needs rebuilding):
        self.T = None
        self.state_space = None
        self.action_space = None

        # Initialize:
        self.update()

    def update(self):

        # Build transitin matrix:
        if self.model=='exhaustive':
            
            # Build state and action space:
            self.action_space = TransitionMatrix.enumerate_actions(env=self.env, as_objects=False, n_selected=self.n_selected)
            self.state_space = TransitionMatrix.enumerate_states(env=self.env, as_objects=False)
            
            # Initialize transition matrix:
            self.T = np.zeros((len(self.action_space), len(self.state_space), len(self.state_space)))

            # Calculate transition probabilities:
            for i, action in enumerate(self.action_space):
                for j, state1 in enumerate(self.state_space):
                    for k, state2 in enumerate(self.state_space):

                        # Check that the new state is consistent with the action alone
                        both_true = action & state2
                        consistent = np.sum(both_true)
                            
                        if consistent == self.n_selected:
                        
                            # Calculate the probabilities of influence occuring to each next state
                            total_influence_prob = []
                            consistency_check = 1

                            for n_state, n_val in enumerate(state2):
                                next_state_prob = []
                                if action[n_state]==True:
                                    next_state_prob.append(0)
                                else:
                                    for c_state, c_val in enumerate(state1):
                                        if c_val == 1:

                                            # Special case where the states are the same
                                            if n_state == c_state: 

                                                # Consistency check 
                                                if n_val == 0:
                                                    consistency_check = 0
                                                    next_state_prob.append(1)
                                                else:
                                                    next_state_prob.append(0)
                                            else:
                                                next_state_prob.append(1 - self.env.influence.matrix[c_state, n_state])
                                        else:
                                            next_state_prob.append(1)

                                prob_no_influence = np.prod(next_state_prob)
                                prob_influence = 1 - prob_no_influence
                                if n_val == 0:
                                    total_influence_prob.append(prob_no_influence)
                                else:
                                    total_influence_prob.append(prob_influence)
                        
                            if consistency_check == 1:
                                total_probability = np.prod(total_influence_prob)
                            else:
                                total_probability = 0
                            self.T[i,j,k] = total_probability

        elif self.model in {'exhaustive_fast','reachable','pruned'}:
            
            # Build state and action space (possibly limiting to meaningful actions and reachable states):
            if self.model=='exhaustive_fast':
                 # Branch from all states (i.e. pass `state=None` to the enumeration function):
                self.action_space = TransitionMatrix.enumerate_actions(env=self.env, state=None, as_objects=False, n_selected=self.n_selected)
                self.state_space = TransitionMatrix.enumerate_states(env=self.env, state=None, as_objects=False)
                starting_states = self.state_space
            elif self.model=='reachable':
                # Subset to reachable states and meaninful actions:
                self.action_space = TransitionMatrix.enumerate_actions(env=self.env, state=self.env.state, as_objects=False, n_selected=self.n_selected)
                self.state_space = TransitionMatrix.enumerate_states(env=self.env, state=self.env.state, as_objects=False)
                starting_states = self.state_space
            elif self.model=='pruned':
                # Only branch from current state:
                self.action_space = TransitionMatrix.enumerate_actions(env=self.env, state=self.env.state, as_objects=False, n_selected=self.n_selected)
                self.state_space = TransitionMatrix.enumerate_states(env=self.env, state=self.env.state, as_objects=False)
                starting_states = [self.env.state.vector]
            # Common to both methods:
            useful_actions = self.action_space
            landing_states = self.state_space

            # Initialize transition matrix:
            self.T = np.zeros((len(useful_actions), len(starting_states), len(landing_states)))
            
            for i, action in enumerate(useful_actions):
                for j, state1 in enumerate(starting_states):

                    # Calculate how likely each agent is to be informed at the end of this step:
                    probs = TransitionMatrix.agent_probabilities(env=self.env, state=state1, action=action)
                    
                    for k, state2 in enumerate(landing_states):
                        
                        # Flip the probability for agents who should end up not informed and
                        # get total probability by multiplying how likely each new state is for each agent:
                        total_probability = np.prod( np.where(state2==True, probs, 1-probs) )
                        
                        # Store result in matrix:
                        self.T[i,j,k] = total_probability

        else:
            raise NotImplementedError(f"Influence model {self.model} is not yet implemented.")

        assert np.all( self.T.sum(axis=-1)==1 ), "Expected all rows to sum to 1."

        return self.T


class Graph:
    """
    A graph of the influence network (using networkx).
    """

    def __init__(self, env):
        """
        env:
            The simulation environment.
        """
        self.env = env

        # Define state variables (initialized below):
        self.G = None
        self.pos = None
        self.edge_labels = None

        # Initialize:
        self.update_structure()
        #self.update_layout()  # Performed on demand by utility functions.

    @property
    def node_labels(self):
        """List of agent_ids, in node order."""
        return list(self.G.nodes)

    @property
    def node_index_lookup(self):
        """Lookup from agent_id to index in node list."""
        return {agent_id:node_index for node_index,agent_id in enumerate(self.G.nodes)}

    @property
    def state_index_lookup(self):
        """Lookup from agent_id to position in state vector."""
        return {agent_id:state_index for state_index,agent_id in enumerate(self.env.agent_ids)}

    def update_structure(self):
        
        # Build graph:
        self.G = nx.DiGraph()

        # Create the Graph structure
        for agent in self.env.agents.values():
            for next_agent_id in agent.inner_circle:
                connection_strength = self.env.influence.matrix[(self.state_index_lookup[agent.id], self.state_index_lookup[next_agent_id])]
                self.G.add_edge(agent.id, next_agent_id, val=connection_strength, edge_color='#DE3D83')

            for next_agent in agent.outer_circle:
                connection_strength = self.env.influence.matrix[(self.state_index_lookup[agent.id], self.state_index_lookup[next_agent_id])]
                self.G.add_edge(agent.id, next_agent, val=connection_strength, edge_color='black')

    def update_layout(self, iterations=None):
        """
        Calculate node positions (with specified number of iterations)
        """

        # Build graph if needed:
        if self.G is None:
            self.update_structure()

        # Apply default value:
        if iterations is None:
            iterations = 20

        # Calculate positions and build edge labels:
        self.pos = nx.spring_layout(self.G, iterations=iterations)
        self.edge_labels = dict([((node1, node2, ), f'{connection_data["val"]}\n\n{self.G.edges[(node2,node1)]["val"]}')
                for node1, node2, connection_data in self.G.edges(data=True) if self.pos[node1][0] > self.pos[node2][0]])

    def plot_network_graph(self, iterations=None, influenced=None, action_nodes=None, figsize=None, ax=None):
        """
        Plot the network graph in the specified axes (or not axes of not specified).
        (The `iterations` parameter is passed to the layout update function, only if needed.)
        Returns the axes.
        """

        # Update layout (and build graph) if needed:
        if (self.pos is None) or (self.edge_labels is None):
            self.update_layout(iterations=iterations)

        if influenced is None:
            influenced = self.env.state.vector  # Boolean vector.

        if action_nodes is None:
            action_nodes = np.zeros(len(self.env.agent_ids), dtype=int)

        # Set up color map
        color_map = []
        for node in self.node_labels:
            state_index = self.state_index_lookup[node]
            if influenced[state_index] and action_nodes[state_index]:
                color_map.append('#dbb700')
            elif influenced[state_index]:
                color_map.append('#f45844')
            elif action_nodes[state_index]:
                color_map.append('#3dbd5d')
            else:
                color_map.append('#0098d8')

        if ax is None:
            if figsize is None:
                figsize = (10,7)
            _, ax = plt.subplots(figsize=figsize)
        
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=self.edge_labels, font_color='red')
        nx.draw_networkx(self.G, self.pos, with_labels=True, node_size=400, node_color=color_map, ax=ax, connectionstyle='arc3, rad = 0.1')

        return ax


class State:

    """
    Static representation of a state of the environment
    (i.e. whether or not each professional is informed).
    """

    @classmethod
    def coerce(self, env, state):
        """
        Coerces the input to a State object.

        Accepts several input formats:
            - a State object.
            - a boolean vector where each value corresponds to an agent in env.agent_ids.
            - a list of informed agent_ids.
            - a dictionary of booleans keyed by agent_ids.

        Note: If a list/array is received, determine whether it is a list of agent_ids or a boolean vector. 
        We assume that if the vector is of the same length as the numer of agents and has only boolean-like
        values, it is a boolean state vector. This might fail un some unlikely edge cases (e.g. there are exactly two agents with IDs 0 and 1) but otherwise should be fairly robust/efficient.
        """
        if isinstance(state, State):
            if env != state.env:
                return State(env=env, vector=state.vector)
            else:
                return state
        elif isinstance(state, dict):
            return State(env=env, lookup=state)
        else:
            bool_values = set([True,False,0,1,0.0,1.0])
            is_boolean = ( set(state) | bool_values ) == bool_values
            is_full_length = (len(state)==len(env.agent_ids))
            state = np.array(state)
            if is_boolean and is_full_length:
                return State(env=env, vector=state)
            else:
                return State(env=env, informed_ids=state)

    def __init__(self, env, vector=None, informed_ids=None, lookup=None):
        """
        An immutable representation of the state (i.e. which agents are informed, in the same order as env.agent_ids)
        env:
            The simulation environment.
        vector:
            A boolean vector of informed status (one value for each agent in env.agent_id order).
        informed_ids:
            A list of agent_ids (or agent objects) who are informed.
        lookup:
            A dict of booleans (keyed by agent_id) indicating which agents are informed.
        Exactly one of the methods (vector, ids, lookup) should be specified.
        Providing a vector is fastest but it is not checked for validity.
        """
        self.env = env
        self.n_agents = len(self.env.agent_ids)
        self._vector = np.repeat(False, self.n_agents)
        self._informed_ids = None  # Built on demand.
        self._lookup = None  # Built on demand.
        if vector is not None:    
            assert len(vector)==self.n_agents
            self._vector = np.array(vector)
        elif informed_ids is not None:
            self._informed_ids = []
            for agent_id in informed_ids:
                try:
                    agent_id = agent_id.id  # If Agent object.
                except:
                    pass  # If integer.
                self._informed_ids.append(agent_id)
                agent_index = self.env.agent_indices[agent_id]
                self._vector[agent_index] = True
        elif lookup is not None:
            self._lookup = {agent_id:False for agent_id in self.env.agent_ids}
            for agent_id, val in lookup:
                agent_index = self.env.agent_indices[agent_id]
                self._lookup[agent_index] = val
                self._vector[agent_index] = val
            assert len(self._lookup) == self.n_agents
        else:
            raise ValueError("Must specify `vector`, `selected_ids`, or `lookup`.")
        self.n_informed = sum(self.vector)

    @property
    def vector(self):
        """
        Return the state as a numpy boolean array in the same order as the adjacency matrix.
        """
        return self._vector

    @property
    def informed_ids(self):
        if self._informed_ids is None:
            self._informed_ids = [agent_id for agent_id,val in zip(self.env.agent_ids,self._vector) if val]
        return self._informed_ids

    @property
    def lookup(self):
        if self._lookup is None:
            self._lookup = {agent_id:val for agent_id,val in zip(self.env.agent_ids,self._vector) if val}
        return self._lookup

    def __str__(self):
        return f"<State with {self.n_informed}/{self.n_agents} informed agents>"

    def __repr__(self):
        # Note: An exact representation would need to include the environment.
        return "State{}".format(list(self.vector))
    
    def copy(self):
        return State(env=self.env, vector=self.vector.copy())


class Action:

    """
    Static representation of an action in the environment
    (i.e. whether or not to intervene for each individual).
    """

    @classmethod
    def coerce(self, env, action):
        """
        Coerces the input to an Action object.

        Accepts several input formats:
            - an Action object.
            - a boolean vector where each value corresponds to an agent in env.agent_ids.
            - a list of selected agent_ids.
            - a dictionary of booleans keyed by agent_ids.

        Note: If a list/array is received, determine whether it is a list of agent_ids or a boolean vector. 
        We assume that if the vector is of the same length as the numer of agents and has only boolean-like
        values, it is a boolean action vector. This might fail un some unlikely edge cases (e.g. there are exactly two agents with IDs 0 and 1) but otherwise should be fairly robust/efficient.
        """
        if isinstance(action, Action):
            if env != action.env:
                return Action(env=env, vector=action.vector)
            else:
                return action
        elif isinstance(action, dict):
            return Action(env=env, lookup=action)
        else:
            bool_values = set([True,False,0,1,0.0,1.0])
            is_boolean = ( set(action) | bool_values ) == bool_values
            is_full_length = (len(action)==len(env.agent_ids))
            action = np.array(action)
            if is_boolean and is_full_length:
                return Action(env=env, vector=action)
            else:
                return Action(env=env, selected_ids=action)

    @classmethod
    def sort_actions(cls, actions, state=None, method='random'):
        """
        Return a list of value,action tuples, sorted in descending order.
        Note, this calls the `get_value` function on each method,
        which could be computationally expensive.

        actions:
            A list of interventions to evaluate.
        state:
            The state for which to evaluate the method.
            (If None, defaults to current environment state.)
        method:
            The evaluation method to use.

        """

        results = []
        for action in actions:
            value = action.get_value(state=state, method=method)
            results.append( (value,action) )
        results = sorted(results, key=lambda pair: pair[0], reverse=True)
        return results

    def __init__(self, env, vector=None, selected_ids=None, lookup=None):
        """
        An immutable representation of the action (i.e. which agents are selected, in the same order as env.agent_ids)
        env:
            The simulation environment.
        vector:
            A boolean vector of selected status (one value for each agent in env.agent_id order).
        selected_ids:
            A list of agent_ids (or agent objects) who are selected.
        lookup:
            A dict of booleans (keyed by agent_id) indicating which agents are selected.
        Exactly one of the methods (vector, ids, lookup) should be specified.
        Providing a vector is fastest but it is not checked for validity.
        """
        self.env = env
        self.n_agents = len(self.env.agent_ids)
        self._vector = np.repeat(False, self.n_agents)
        self._selected_ids = None  # Built on demand.
        self._lookup = None  # Built on demand.
        if vector is not None:    
            assert len(vector)==self.n_agents
            self._vector = np.array(vector)
        elif selected_ids is not None:
            self._selected_ids = []
            for agent_id in selected_ids:
                try:
                    agent_id = agent_id.id  # If Agent object.
                except:
                    pass  # If integer.
                self._selected_ids.append(agent_id)
                agent_index = self.env.agent_indices[agent_id]
                self._vector[agent_index] = True
        elif lookup is not None:
            self._lookup = {agent_id:False for agent_id in self.env.agent_ids}
            for agent_id, val in lookup:
                agent_index = self.env.agent_indices[agent_id]
                self._lookup[agent_index] = val
                self._vector[agent_index] = val
            assert len(self._lookup) == self.n_agents
        else:
            raise ValueError("Must specify `vector`, `selected_ids`, or `lookup`.")
        self.n_selected = sum(self.vector)

    @property
    def vector(self):
        """
        Return the action as a numpy boolean array in the same order as the adjacency matrix.
        """
        return self._vector

    @property
    def selected_ids(self):
        if self._selected_ids is None:
            self._selected_ids = [agent_id for agent_id,val in zip(self.env.agent_ids,self._vector) if val]
        return self._selected_ids

    @property
    def lookup(self):
        if self._lookup is None:
            self._lookup = {agent_id:val for agent_id,val in zip(self.env.agent_ids,self._vector) if val}
        return self._lookup

    def __str__(self):
        return f"<Action with {self.n_selected}/{self.n_agents} selected agents>"

    def __repr__(self):
        # Note: An exact representation would need to include the environment.
        return "Action{}".format(list(self.vector))
    
    def copy(self):
        return Action(env=self.env, vector=self.vector.copy())

    def get_value(self, state=None, method='random'):
        if method=='random':  # For testing.
            value = self.env.random.uniform(10)
            value = np.round(value, 1)
        else:
            raise NotImplementedError(f"Estimation method {method} is not defined.")
        return value


class Environment:

    """
    A simulation of information propagation in a network of healthcare professionals.
    The state of the system is determine by whether or not each professional is **informed**.
    The actions are defined by which combination of professionals is **selected** for intervention.
    """

    def __init__(
        self, agents, agent_ids=None, seed=None,
        base_receptivity = 0.5,
        base_persuasiveness = 0.5,
        intervention_size = 100,
        influence_model = None,
        transition_model = None,
    ):
        """
        agents:
            A list or dict (keyed by agent_id) of Agent objects.
        agent_ids:
            (Optional) List of agent_ids in the order they should be stored in the matrix.
            If not provided, defaults to sorted order of agent_ids.
        seed:
            (Optional) An integer seed for numpy random state.
        
        HYPERPARAMETERS:

        base_receptivity:
            (float between 0.0 and 1.0)
            How receptive agents are when receiving information for neighbors.
            This value is only used as the default for agents that were initialized without a value.
        base_persuasiveness:
            (float between 0.0 and 1.0)
            How presuasive agents are when transimitting information for neighbors.
            This value is only used as the default for agents that were initialized without a value.
        intervention_size:
            (int)
            The default number of agents to select for intervention at each step.
            (Under some transition models, actions may include fewer interventions if
            the intervention_size is larger than the number of uninformed agents.)
        influence_model:
            (string: 'default')
            [Defaults are handled in the InfluenceMatrix class.]
        transition_model:
            (string: 'exhaustive', 'exhaustive_fast', 'pruned')
            [Defaults are handled in the TransitionMatrix class.]
        """

        # Hyperparameters:
        self.base_receptivity = base_receptivity
        self.base_persuasiveness = base_persuasiveness
        self.intervention_size = intervention_size
        self.influence_model = influence_model
        self.transition_model = transition_model

        # Random state:
        self.seed = seed
        self.random = np.random.RandomState(seed)

        # Network structure:
        self.agents = Agent.to_dict(agents)
        self.workplaces = dict()
        self.specialties = dict()
        
        # Maintain a list of agent_ids in the order the appear in the matrix:
        self.agent_ids = agent_ids if agent_ids else list(sorted(self.agents.keys()))
        self.agent_indices = {agent_id:agent_index for agent_index,agent_id in enumerate(self.agent_ids)}

        # Network state (set by update function):
        self.state = None  # Boolean indicators of which agents are informed.
        self.inner = None  # AdjacnecyMatrix for inner circle networks.
        self.outer = None  # AdjacnecyMatrix for outer circle networks.
        self.influence = None  # InfluenceMatrix.

        # Networks Graph
        self.graph = None

        # Transition Matrix
        self.trans = None

        # Initialize:
        self.update()

    def update(self):
        """
        Peform all environment updates.
        """
        self.update_agents()
        self.update_workplaces()
        self.update_specialties()
        self.update_adjacency()
        self.update_influence()
        self.update_state()
        #self.update_network_graph()
        #self.update_transition_matrix()

    def update_agents(self):
        """
        Bind each agent to environment.
        """
        for agent_id, agent in self.agents.items():
            agent.env = self
            # Apply defaults only for agent values that are not already set:
            if agent.informed_init is None:
                agent.informed_init = False
            if agent.informed is None:
                agent.informed = agent.informed_init
            if agent.receptivity is None:
                agent.receptivity = self.base_receptivity
            if agent.persuasiveness is None:
                agent.persuasiveness = self.base_persuasiveness

    def update_workplaces(self):
        """
        Rebuild workplace lookup.
        """
        for agent_id, agent in self.agents.items():
            for workplace_id in agent.workplace_ids:
                if workplace_id not in self.workplaces:
                    self.workplaces[workplace_id] = []
                self.workplaces[workplace_id].append(agent)

    def update_specialties(self):
        """
        Rebuild speciality lookup.
        """
        for agent_id, agent in self.agents.items():
            for specialty_id in agent.specialty_ids:
                if specialty_id not in self.specialties:
                    self.specialties[specialty_id] = []
                self.specialties[specialty_id].append(agent)

    def update_adjacency(self):
        """
        Rebuild agent adjacency matrix.
        """
        self.inner = AdjacencyMatrix(self, scope='inner', agent_ids=self.agent_ids)
        self.outer = AdjacencyMatrix(self, scope='outer', agent_ids=self.agent_ids)

    def update_influence(self, model=None):
        """
        Rebuild agent adjacency matrix.
        """
        model = model if model is not None else self.influence_model
        self.influence = InfluenceMatrix(self, model=model, agent_ids=self.agent_ids)

    def update_state(self, new_state=None):
        """
        Build state object to reflect current agent states.
        new_state:
            A new state to update the agent states to.
            (Optional -- if not provided, agent states are left unchanged.)
        """

        # Apply specified state, if provided (otherwise keep current state):
        if new_state is not None:
            # Coerce to state object:
            new_state = State.coerce(env=self, state=new_state)
            # Update agents in the environment to reflect requested state:
            for agent_id, val in zip(self.agent_ids, new_state.vector):
                self.agents[agent_id].informed = val
        else:
            new_state = [self.agents[agent_id].informed for agent_id in self.agent_ids]
            new_state = State(env=self, vector=new_state)

        # Update state (i.e. which agents are informed):
        self.state = new_state

    def update_transition_matrix(self, n_selected=None, model=None):
        """
        WARNING - Only run for small problem sizes

        n_selected:
            Number of agents being selected for intervention at each timestep.
            If None, defaults to self.intervention_size.
        model:
            The model used to build the transition matrix ('exhaustive' or 'pruned').
        """
        model = model if model is not None else self.transition_model
        self.trans = TransitionMatrix(env=self, model=model, n_selected=n_selected)

    def update_network_graph(self):
        """
        A graph representation of the agent influence network.
        """
        # Initialize the graph
        self.graph = Graph(env=self)

    def plot_network_graph(self, iterations=None, influenced=None, action_nodes=None, figsize=None, ax=None):
        """
        Draw the network graph.
        This is a wrapper function for Graph.plot_network_graph,
        which calculates positions if needed.
        """
        # Make sure the graph object has been initialized:
        if self.graph is None:
            self.update_network_graph()
        # Pass parameters through to Graph.plot_network_graph, which updates as needed:
        return self.graph.plot_network_graph(iterations=iterations, influenced=influenced, action_nodes=action_nodes, figsize=figsize, ax=ax)

    @property
    def G(self):
        """
        Alias for the network graph property.
        """
        if self.graph is None:
            return None
        return self.graph.G

    @property
    def T(self):
        """
        Alias for the transition matrix.
        """
        if self.trans is None:
            return None
            #raise RuntimeError("self.update_transition_matrix has not been called.")
        return self.trans.T

    @property
    def state_space(self):
        if self.trans is None:
            return None
            #raise RuntimeError("self.update_transition_matrix has not been called.")
        return self.trans.state_space

    @property
    def action_space(self):
        if self.trans is None:
            return None
            #raise RuntimeError("self.update_transition_matrix has not been called.")
        return self.trans.action_space

    @property
    def n_informed(self):
        return self.state.n_informed

    def __str__(self):
        return f"<Environment with {self.state.n_informed}/{self.state.n_agents} informed agents>"
