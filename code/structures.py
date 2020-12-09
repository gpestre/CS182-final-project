"""
Basic data structures for modeling information propagation.
"""

import warnings
import numpy as np
import scipy.sparse
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools

from matplotlib.colors import Normalize as colorNormalize
from matplotlib.lines import Line2D


class Agent:
    
    all_agents = list()
    
    @classmethod
    def reset(cls):
        print("WARNING: Agent class was reset, which may cause agent_id conflicts with other simulations.")
        cls.all_agents = list()

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
        self._id = len(Agent.all_agents)
        Agent.all_agents.append(self)
        self.id = self._id

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

    def copy(self):
        agent = Agent(
            env = self.env,
            workplace_ids = None if self.workplace_ids is None else self.workplace_ids.copy(),
            specialty_ids = None if self.specialty_ids is None else self.specialty_ids.copy(),
            inner_circle = None if self.inner_circle is None else self.inner_circle.copy(),
            outer_circle = None if self.outer_circle is None else self.outer_circle.copy(),
            informed_init = self.informed_init,
            receptivity = self.receptivity,
            persuasiveness = self.persuasiveness,
        )
        agent.id = self.id
        agent.informed = self.informed
        agent.intervention = self.intervention
        return agent


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
    def enumerate_actions(cls, env, n_selected, state=None, as_objects=False):
        """
        Enumerate all possible inteventions in a given environment.
        env:
            The simulation environment.
        n_selected:
            (int) The max number of inteventions (capped at number of agents).
        state:
            (State object) If specified, enumerates only actions that can be meaningfully taken at this state.
            If False, exhaustively list all possible actions (including those that select already informed agents).
        as_objects:
            (bool) Optionally return a list of Action objects.
            Otherwise, return a list of boolean arrays (where values are in the same order as env.agent_ids).
        """
        # Get intervention size:
        assert n_selected is not None, f"This helper method expects n_selected to be specified explicitly."
        assert n_selected >= 0
        n_selected = min(n_selected, len(env.agent_ids))  # Limit to number of agents.
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
            # Get list of reachable states for each agent:
            agent_states = [[True] if s else [False,True] for s in current_state.vector]
            # Build list of reachable states (all possible combinations of agent-states):
            states = [np.array(state) for state in itertools.product(*agent_states)]
        if as_objects:
            # Build an action object (using a dict of booleans):
            states = [
                State(env=env, vector=state)
                for state in states
            ]
        return states

    @classmethod
    def agent_probabilities(cls, env, state, action=None):
        """
        Given a starting state and an action, returns the probability that each agent is informed (in the landing state).
        The return is a vector of floats, corresponding to the agents in env.agent_id order.
        env:
            The simulation environment.
        state:
            The starting state.
        action:
            The applied action (or None to get only the effect of organic transtion).
        """

        # Get boolean representation of state:
        state = State.coerce(env=env, state=state).vector  # Vector of booleans.

        # Get state vector and influence matrix:
        probs = env.influence.matrix

        # Convert to numpy matrix (won't remain sparse during manipulation):
        probs = probs.toarray()

        # Multiply by state column -- uninformed agents will not influence anyone:
        probs = np.where(state.reshape(-1,1), probs, 0)

        # Multiply by state row -- already informed agents will remain informed:
        probs = np.where(state.reshape(1,-1), 1, probs)

        # Apply action if applicable:
        if action is not None:

            # Get boolean representation of action:
            action = Action.coerce(env=env, action=action).vector  # Vector of booleans.

            # Multiply by action row -- agents selected for intervention will be informed:
            probs = np.where(action.reshape(1,-1), 1, probs)

        # Calculate how likely each agent is to be informed at the end of this step:
        probs = 1-np.prod(1-probs,axis=0)

        return probs

    def __init__(self, env, starting_state=None, model=None, n_selected=None, agent_ids=None):
        """
        env:
            The simulation environment.
        model:
            Hyperparameter to control which model is used.
            The 'exhaustive' mode enumerates all states and actions (using loops).
            The 'exhaustive_fast' mode enumerates all states and actions (using matrix operations).
            The 'reachable' mode limits enumeration to meaningful actions, from reachable states to reachable states.
            The 'pruned' mode limits enumeration to meaningful actions, from current state to reachable states.
        starting_state:
            (Optional) The state from which to build this TransitionMatrix.
            This parameter is ignored for exhaustive models.
            For models that use this parameter, the current environment state is used if none is specified.
        n_selected:
            The (max) number of agents to select for each intervention.
        agent_ids:
            (Optional) List of agent_ids in the order they should be stored in the state/acton space.
            If not provided, defaults to the order used by the environmnet.
        """
        self.env = env
        self.starting_state = env.state if starting_state is None else State.coerce(env=env, state=starting_state)
        # Set influence model:
        self.valid_models = {'exhaustive','exhaustive_fast','reachable','pruned'}
        self.model = model if model is not None else 'exhaustive_fast'
        assert self.model in self.valid_models, f"{model} is not a valid transition model: {self.valid_models}"

        # Maintain a list of agent_ids in the order the appear in the matrix:
        self.agent_ids = agent_ids if (agent_ids is not None) else list(sorted(env.agents.keys()))

        # Get intervention size:
        n_selected = env.intervention_size if n_selected is None else int( n_selected )
        n_selected = min(n_selected, len(self.env.agent_ids))
        assert n_selected >= 0
        self.n_selected = n_selected

        # Keep track of matrix in scipy.sparse.csr_matrix (or None when it needs rebuilding):
        self.T = None
        self.landing_states = None  # Aliased by landing_states.
        self.action_space = None
        self.starting_states = None  # May or may not be the full action space -- depends on model.
        self.state_index_lookup = None
        self.action_index_lookup = None

        # Initialize:
        self.update()

    @property
    def state_space(self):
        return self.landing_states

    @state_space.setter
    def state_space(self, state_space):
        self.landing_states = state_space

    def encode_state(self, state_vector):
        state_vector = State.coerce(self.env, state=state_vector).vector
        return self.state_index_lookup[tuple(state_vector)]

    def decode_state(self, state_index):
        return self.state_space[state_index]

    def encode_action(self, action_vector):
        action_vector = Action.coerce(self.env, action=action_vector).vector
        return self.action_index_lookup[tuple(action_vector)]

    def decode_action(self, action_index):
        return self.action_space[action_index]

    def update(self):

        # Build transitin matrix:
        if self.model=='exhaustive':
            
            # Build state and action space:
            self.action_space = TransitionMatrix.enumerate_actions(env=self.env, as_objects=False, n_selected=self.n_selected)
            self.landing_states = TransitionMatrix.enumerate_states(env=self.env, as_objects=False)
            self.starting_states = self.landing_states

            # Add null action:
            null_action = Action(env=self.env, selected_ids=[]).vector
            self.action_space = [null_action] + self.action_space
            
            # Initialize transition matrix:
            self.T = np.zeros((len(self.action_space), len(self.starting_states), len(self.landing_states)))

            # Calculate transition probabilities:
            for i, action in enumerate(self.action_space):
                for j, state1 in enumerate(self.starting_states):
                    for k, state2 in enumerate(self.landing_states):

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
                self.landing_states = TransitionMatrix.enumerate_states(env=self.env, state=None, as_objects=False)
                self.starting_states = self.landing_states
            elif self.model=='reachable':
                # Subset to reachable states and meaninful actions:
                self.action_space = TransitionMatrix.enumerate_actions(env=self.env, state=self.starting_state, as_objects=False, n_selected=self.n_selected)
                self.landing_states = TransitionMatrix.enumerate_states(env=self.env, state=self.starting_state, as_objects=False)
                self.starting_states = self.landing_states
            elif self.model=='pruned':
                # Only branch from current state:
                self.action_space = TransitionMatrix.enumerate_actions(env=self.env, state=self.starting_state, as_objects=False, n_selected=self.n_selected)
                self.landing_states = TransitionMatrix.enumerate_states(env=self.env, state=self.starting_state, as_objects=False)
                self.starting_states = [self.starting_state.vector]

            # Add null action:
            null_action = Action(env=self.env, selected_ids=[]).vector
            self.action_space = [null_action] + self.action_space

            # Initialize transition matrix:
            self.T = np.zeros((len(self.action_space), len(self.starting_states), len(self.landing_states)))
            
            for i, action in enumerate(self.action_space):
                for j, state1 in enumerate(self.starting_states):

                    # Calculate how likely each agent is to be informed at the end of this step:
                    probs = TransitionMatrix.agent_probabilities(env=self.env, state=state1, action=action)
                    
                    for k, state2 in enumerate(self.landing_states):
                        
                        # Flip the probability for agents who should end up not informed and
                        # get total probability by multiplying how likely each new state is for each agent:
                        total_probability = np.prod( np.where(state2==True, probs, 1-probs) )
                        
                        # Store result in matrix:
                        self.T[i,j,k] = total_probability

        else:
            raise NotImplementedError(f"Influence model {self.model} is not yet implemented.")

        sum_check = self.T.sum(axis=-1)
        assert np.allclose( sum_check, np.ones(sum_check.shape), atol=1e-5), "Expected all rows to sum to 1."

        # Create reverse lookups:
        self.state_index_lookup = {tuple(state_vector):state_index for state_index,state_vector in enumerate(self.state_space)}
        self.action_index_lookup = {tuple(action_vector):action_index for action_index,action_vector in enumerate(self.action_space)}

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
        self._edge_colors = None

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
        self._edge_colors = []

        # Create the Graph structure
        for agent in self.env.agents.values():
            for next_agent_id in agent.inner_circle:
                connection_strength = self.env.influence.matrix[(self.state_index_lookup[agent.id], self.state_index_lookup[next_agent_id])]
                self.G.add_edge(agent.id, next_agent_id, val=connection_strength)
                self._edge_colors.append('#0f0a01')

            for next_agent_id in agent.outer_circle:
                connection_strength = self.env.influence.matrix[(self.state_index_lookup[agent.id], self.state_index_lookup[next_agent_id])]
                self.G.add_edge(agent.id, next_agent_id, val=connection_strength)
                self._edge_colors.append('#a5a6a9')

    def update_layout(self, labels, iterations=None, seed=182):
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
        self.pos = nx.spring_layout(self.G, iterations=iterations, seed=seed)
        self.edge_labels = dict()
        if labels == True:
            for node1, node2, connection_data in self.G.edges(data=True):
                if self.pos[node1][0] > self.pos[node2][0]:
                    try:
                        self.edge_labels[(node1,node2)] = f'{connection_data["val"]}\n\n{self.G.edges[(node2,node1)]["val"]}'
                    except:
                        pass


    def plot_network_graph(self, iterations=None, influenced=None, action_nodes=None, labels=True, legend=False, 
                           colors="influence", seed=182, figsize=None, rebuild=False, ax=None):
        """
        Plot the network graph in the specified axes (or not axes of not specified).
        (The `iterations` parameter is passed to the layout update function, only if needed.)
        Returns the axes.
        """

        # Update layout (and build graph) if needed:
        if (self.pos is None) or (self.edge_labels is None) or (self._edge_colors is None) or (rebuild==True):
            self.update_layout(iterations=iterations, labels=labels, seed=seed)

        if influenced is None:
            influenced = self.env.state.vector  # Boolean vector.

        if action_nodes is None:
            action_nodes = np.zeros(len(self.env.agent_ids), dtype=int)

        # Set up color map
        color_map = []
        if colors == "influence":
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
        elif colors == "workplace":
            all_workplaces = list(self.env.workplaces.keys())
            norm = colorNormalize(vmin=min(all_workplaces), vmax=max(all_workplaces)+1, clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
            for node in self.node_labels:
                state_index = self.state_index_lookup[node]
                workplace = self.env.agents[state_index].workplace_ids[0]
                color_map.append(mapper.to_rgba(workplace))
        else:
            raise ValueError("Colors must be one of ['influence', 'workplace']")



        if ax is None:
            if figsize is None:
                figsize = (10,7)
            _, ax = plt.subplots(figsize=figsize)
        
        if labels == True:
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=self.edge_labels, font_color='red')
        nx.draw_networkx(self.G, self.pos, with_labels=True, node_size=400, node_color=color_map, 
                         edge_color=self._edge_colors, ax=ax, connectionstyle='arc3, rad = 0.1')

        if legend == True:
            if colors == "influence":
                legend = [Line2D([0], [0], color="#0f0a01", lw=2, label="Inner Connection"),
                Line2D([0], [0], color="#a5a6a9", lw=2, label="Outer Connection"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor="#f45844", 
                        markersize=15, label="Influenced"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor="#3dbd5d", 
                        markersize=15, label="Chosen Action"),
                Line2D([0], [0], marker='o', color='w', markerfacecolor="#0098d8", 
                        markersize=15, label="Uninfluenced Agent")]
            elif colors == "workplace":
                legend = []
                for workplace_id, color in zip(all_workplaces, set(color_map)):
                    legend.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                        markersize=15, label=f"Workplace: {workplace_id}"))
            else:
                raise ValueError("Colors must be one of ['influence', 'workplace']")
            ax.legend(handles=legend)
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
        if hasattr(state, 'vector') and hasattr(state, 'n_informed'):  # State object.
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
            assert len(vector)==self.n_agents, f"State constructor got {len(vector)} values, expected {self.n_agents}"
            self._vector = np.array(vector, dtype=bool)
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
        if hasattr(action, 'vector') and hasattr(action, 'n_selected'):  # Action object.
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
            assert len(vector)==self.n_agents, f"Action constructor got {len(vector)} values, expected {self.n_agents}"
            self._vector = np.array(vector, dtype=bool)
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


class Policy:
    """
    Superclass for various kinds of policies.
    """
    
    def __init__(self, env):
        self.env = env

    @property
    def action_space(self):
        """
        Return a list of available actions.
        """
        raise NotImplementedError  # Policy superclass.

    def get_action(self, state):
        """
        Recomend action based on policy.
        """
        raise NotImplementedError  # Policy superclass.

class RandomPolicy(Policy):
    """
    Randomly selects agents to inform (even if they are already informed).
    """
    
    def __init__(self, env, n_selected=None):
        """
        Build a policy that randomly selects agents to inform.
        env:
            The simulation environment.
        n_selected:
            The number of agents to select in each action (defaults to env.intervention_size).
        """
        self.env = env

        # Get intervention size:
        n_selected = env.intervention_size if n_selected is None else int( n_selected )
        assert n_selected >= 0
        self.n_selected = n_selected

        # Placeholder for policy properties (initalized below):
        self._action_space = None
        self.n_actions = None
        
        # Initialize:
        self.update()

    def update(self):
        action_space = TransitionMatrix.enumerate_actions(self.env, n_selected=self.n_selected, state=None, as_objects=False)
        if len(action_space)==0:
            action_space = Action(env=self.env, selected_ids=[]).vector
        self._action_space = action_space
        self.n_actions = len(action_space)

    @property
    def action_space(self):
        return self._action_space
    
    def get_action(self, state=None):
        """
        Provde a random action.
        (Ignores `state`, but accepts it for compatibility.)
        """
        index = self.env.random.randint( self.n_actions )
        return self.action_space[index]

class RandomUsefulPolicy(Policy):
    """
    Randomly selects agents to inform (among those who are not informed already).
    """
    
    def __init__(self, env, n_selected=None):
        """
        Build a policy that randomly selects agents to inform.
        env:
            The simulation environment.
        n_selected:
            The number of agents to select in each action (defaults to env.intervention_size).
        """
        self.env = env

        # Get intervention size:
        n_selected = env.intervention_size if n_selected is None else int( n_selected )
        assert n_selected >= 0
        self.n_selected = n_selected

    @property
    def action_space(self):
        raise ValueError("RandomUsefulPolicy does not have a fixed action space -- depends on current state.")

    def get_action_space(self, state):
        """
        Get action space for specified state.
        """
        action_space = TransitionMatrix.enumerate_actions(self.env, n_selected=self.n_selected, state=state, as_objects=False)
        if len(action_space)==0:
            action_space = Action(env=self.env, selected_ids=[]).vector  # Null action.
        return action_space
    
    def get_action(self, state=None):
        """
        Recomend action based on policy, for given state.
        If no state is specified, uses current environment state.
        """
        # Check input:
        state = self.env.state if state is None else state
        if state is None:
            raise ValueError("RandomUsefulPolicy cannot be run without a state.")
        # Get current action space:
        action_space = self.get_action_space(state=state)
        # Choose random action:
        index = self.env.random.randint( len(action_space) )
        return action_space[index]
        
class PolicyIteration(Policy):
    
    def __init__(self, env, trans=None, gamma=None, epsilon=None, max_steps=None, progress=None):
        """
        env:
            The simulation environment.
        trans:
            (TransitionMatrix object)
            If not specified, uses the environment's transition matrix.
        gamma:
            Discount factor.
        epsilon:
            Convergence tolerance.
        max_steps:
            Max updates for value iteration.
        progress:
            How often to print progress (or None).
        """
        self.env = env
        self.trans = self.env.trans if trans is None else trans
        # Make sure transition matrix has been udpated:
        if self.trans is None:
            raise RuntimeError("env.build_transition_matrix was never called.")

        self.updated = False

        # Check starting and landing states:
        if len(self.starting_states) != len(self.landing_states):
            print(self.starting_states)
            print(self.landing_states)
            raise ValueError("Policy iteration requires same number of starting and landing states.")

        # Solver properties:
        self.gamma = 0.85 if gamma is None else gamma
        self.epsilon = 0.01 if epsilon is None else epsilon
        self.max_steps = float("+Inf") if max_steps is None else max_steps
        self.progress = 20 if progress is None else progress
        
        # Solver variables:
        self.values = self.initialize_values()
        self.rewards = self.initialize_rewards()
        self.policy = self.initialize_policy()

        # Initialize:
        self.update()

    @property
    def starting_states(self):
        return self.trans.starting_states

    @property
    def landing_states(self):
        return self.trans.landing_states

    @property
    def state_space(self):
        return self.trans.state_space

    @property
    def action_space(self):
        return self.trans.action_space

    @property
    def T(self):
        return self.trans.T

    def encode_state(self, state_vector):
        return self.trans.encode_state(state_vector)

    def decode_state(self, state_index):
        return self.trans.decode_state(state_index)

    def encode_action(self, action_vector):
        return self.trans.encode_action(action_vector)

    def decode_action(self, action_index):
        return self.trans.decode_action(action_index)

    def initialize_values(self):
        """
        Value array for states (1D)
        """
        return np.zeros(len(self.state_space))

    def initialize_rewards(self):
        """
        Create a R(s,'s) matrix (reward can be thought of as independent of action)
        The reward is equal to the increase in the number of agents influenced
        """
        R = np.zeros((len(self.starting_states), len(self.landing_states)))
        for i, state1 in enumerate(self.starting_states):
            for j, state2 in enumerate(self.landing_states):
                reward = np.max((0, (np.sum(state2) - np.sum(state1))))
                R[i,j] = reward
        return R

    def initialize_policy(self):
        """
        Value array for states (1D)
        """
        return np.zeros(len(self.starting_states)).astype(int)

    def calculate_policy_value(self, policy, values=None, rewards=None, gamma=None, epsilon=None, max_steps=None, progress=None):
        """
        Perform policy estimation.
        """

        # Get default values:
        gamma = self.gamma if gamma is None else gamma
        epsilon = self.epsilon if epsilon is None else epsilon
        max_steps = self.max_steps if max_steps is None else max_steps
        progress = self.progress if progress is None else progress
        policy = self.policy if policy is None else policy
        values = self.values if values is None else values
        rewards = self.rewards if rewards is None else rewards

        # Evaluate:
        new_values = values.copy()
        counter = 0
        while counter < max_steps:

            deltas = []
            for state_index, state in enumerate(self.state_space):
                
                # Extract the values relevant to the current state
                cur_value = new_values[state_index]
                cur_action_index = policy[state_index]
                
                transition_matrix = self.T[cur_action_index,state_index,:]
                reward_matrix = rewards[state_index,:].reshape(-1,)

                # Calculate the next value using Bellman update
                next_value = np.matmul(transition_matrix, (reward_matrix + gamma * new_values))
                
                # Update the value matrix 
                new_values[state_index] = next_value
                deltas.append(abs(next_value - cur_value))

            counter += 1
            if isinstance(progress, int) and (progress>0) and (counter % progress == 0):
                print(f"{counter} iterations run - max delta = {np.max(deltas)}")
            if np.max(deltas) < epsilon:
                break

        return new_values

    def calculate_policy_improvement(self, policy, values=None, rewards=None, gamma=None):
        """
        Perform policy improvement.
        """

        # Get default values:
        gamma = self.gamma if gamma is None else gamma
        policy = self.policy if policy is None else policy
        values = self.values if values is None else values
        rewards = self.rewards if rewards is None else rewards
    
        # Evaluate:
        stable_policy = True
        new_policy = policy.copy()
        for state_index, state in enumerate(self.state_space):

            old_policy = policy.copy()

            # Calculate the value of taking a specific action followed by the
            # original policy
            action_values = []
            for action_index, action in enumerate(self.action_space):
                action_value = np.matmul(self.T[action_index, state_index, :], 
                                        (rewards[state_index,:] + gamma * values))
                action_values.append(action_value)
            
            best_action = np.argmax(action_values)

            # Update the policy
            new_policy[state_index] = best_action

            if new_policy[state_index] != old_policy[state_index]:
                stable_policy = False

        return new_policy, stable_policy

    def policy_iteration(self, policy, values=None, rewards=None, gamma=None, epsilon=None, max_steps=None, progress=None):
        """
        Perform policy iteration.
        """
        # Get default values:
        gamma = self.gamma if gamma is None else gamma
        epsilon = self.epsilon if epsilon is None else epsilon
        max_steps = self.max_steps if max_steps is None else max_steps
        progress = self.progress if progress is None else progress
        policy = self.policy if policy is None else policy
        values = self.values if values is None else values
        rewards = self.rewards if rewards is None else rewards

        # Evaluate:
        stable = False
        new_policy = policy.copy()
        new_values = values.copy()
        while stable == False:
            new_values = self.calculate_policy_value(
                policy = new_policy, values = new_values, rewards = rewards, gamma = gamma,
                epsilon=epsilon, max_steps=max_steps, progress=progress
            )
            new_policy, stable = self.calculate_policy_improvement(
                policy = new_policy, values = new_values, rewards = rewards, gamma = gamma,
            )

        return new_policy     

    def update(self):
        """
        Recalculate policy.
        """
        self.policy = self.policy_iteration(policy=self.policy, values=self.values, rewards=self.rewards)
        # Set flag:
        self.updated = True

    def get_action(self, state):
        """
        Recomend action based on policy.
        """
        # Check flag:
        if not self.updated:
            raise RuntimeError("Policy has not been updated.")
        state_index = self.encode_state(state_vector=state)
        action_index = self.policy[state_index]
        return self.action_space[action_index]

class DegreeCentrality(Policy):
    """
    Selects the agent which is most highly connected (and currently uninfluenced)

    Tiebreaker is the lowest action index for reproducibility
    """
    
    def __init__(self, env, n_selected=None):
        """
        Build a policy that randomly selects agents to inform.
        env:
            The simulation environment.
        n_selected:
            The number of agents to select in each action (defaults to env.intervention_size).
        """
        self.env = env

        # Get intervention size:
        n_selected = env.intervention_size if n_selected is None else int( n_selected )
        assert n_selected >= 0
        self.n_selected = n_selected

    def get_action_space(self, state=None):
        """
        Return only agents which are not currently influenced
        """
        state = self.env.state if state is None else state
        if state is None:
            raise ValueError("RandomPolicy cannot be run without a state.")

        action_space = TransitionMatrix.enumerate_actions(self.env, n_selected=self.n_selected, state=state, as_objects=False)
        if len(action_space)>0:
            return action_space
        else:
            return Action(env=self.env, selected_ids=[]).vector  # Null action.

    def get_action(self, state=None):
        """
        Recommend action based on policy.
        """
        action_space = self.get_action_space(state=state)

        # Calculate the centrality score fore ach possible action
        centrality = dict()
        for aid, agent in self.env.agents.items():
            centrality[aid] = len(agent.inner_circle) + len(agent.outer_circle)

        action_centralities = []
        for possible_action in action_space:
            action_centrality = 0
            for i, agent in enumerate(possible_action):
                if agent:
                    action_centrality += centrality[i]
            action_centralities.append(action_centrality)

        # Select optimal based on the maximum centrality
        optimal_action_index = np.argmax(action_centralities)
        return action_space[optimal_action_index]

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
        policy_model = None,
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
            (string: 'exhaustive', 'exhaustive_fast', 'reachable', 'pruned')
            [Defaults are handled in the TransitionMatrix class.]
        policy_model:
            (string: 'policy_evaluation', 'random_useful_policy', 'random_policy')
            [Defaults are handled in the Policy class.]
        """

        # Hyperparameters:
        self.base_receptivity = base_receptivity
        self.base_persuasiveness = base_persuasiveness
        self.intervention_size = intervention_size
        self.influence_model = influence_model
        self.transition_model = transition_model
        self.policy_model = policy_model

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

        # Policy
        self.policy = None

        # Simulation state (updated at the beginning of each simulation step):
        self.step_count = None
        self.state_history = None
        self.action_history = None

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
        #self.build_network_graph()
        #self.build_transition_matrix()
        #self.build_policy()
        self.reset_simulation()

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
            # Make sure inner circle is a list of agent_ids not objects:
            inner_circle = []
            for other_agent_id in agent.inner_circle:
                try:
                    other_agent_id = other_agent_id.id
                except:
                    pass
                inner_circle.append(other_agent_id)
            agent.inner_circle = sorted(set(inner_circle))
            # Make sure inner circle is a list of agent_ids not objects:
            outer_circle = []
            for other_agent_id in agent.outer_circle:
                try:
                    other_agent_id = other_agent_id.id
                except:
                    pass
                outer_circle.append(other_agent_id)
            agent.outer_circle = sorted(set(outer_circle))

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

    def build_transition_matrix(self, starting_state=None, n_selected=None, model=None):
        """
        WARNING - Only run for small problem sizes

        starting_state:
            The state from which to branch (not applicable for exhaustive models).
        
        n_selected:
            Number of agents being selected for intervention at each timestep.
            If None, defaults to self.intervention_size.
        model:
            The model used to build the transition matrix ('exhaustive' or 'pruned').
        """
        model = model if model is not None else self.transition_model
        n_selected = self.intervention_size if n_selected is None else n_selected
        self.trans = TransitionMatrix(env=self, starting_state=starting_state, model=model, n_selected=n_selected)

    def build_policy(self, model=None, *policy_args, **policy_kwargs):
        """
        model:
            The model used to evaluate the policy.
        """
        model = model if model is not None else self.policy_model
        valid_models = {'policy_iteration','random_useful_policy','random_policy', 'degree_centrality'}
        assert model in valid_models, f"{model} is not a valid model : {valid_models}"
        if model=='policy_iteration':
            self.policy = PolicyIteration(env=self, *policy_args, **policy_kwargs)
        elif model=='random_useful_policy':
            self.policy = RandomUsefulPolicy(env=self, *policy_args, **policy_kwargs)
        elif model=='random_policy':
            self.policy = RandomPolicy(env=self, *policy_args, **policy_kwargs)
        elif model=='degree_centrality':
            self.policy = DegreeCentrality(env=self, *policy_args, **policy_kwargs)
        else:
            raise NotImplementedError("Policy model {model} is not yet implemented.")

    def build_network_graph(self):
        """
        A graph representation of the agent influence network.
        """
        # Initialize the graph
        self.graph = Graph(env=self)

    def plot_network_graph(self, iterations=None, influenced=None, action_nodes=None, labels=True, legend=False, 
                           colors="influence", seed=182, figsize=None, rebuild=False, ax=None):
        """
        Draw the network graph.
        This is a wrapper function for Graph.plot_network_graph,
        which calculates positions if needed.
        """
        # Make sure the graph object has been initialized:
        if self.graph is None:
            self.build_network_graph()
        # Pass parameters through to Graph.plot_network_graph, which updates as needed:
        return self.graph.plot_network_graph(iterations=iterations, influenced=influenced, action_nodes=action_nodes, labels=labels, legend=legend,
                                             colors=colors, seed=seed, figsize=figsize, rebuild=rebuild, ax=ax)

    def reset_simulation(self):
        """
        Clear state and action histories.
        """
        self.step_count = 0
        self.state_history = []
        self.action_history = []

    def simulate_steps(self, n_steps=1, dry_run=False, seed=None):
        """
        Simulate one step of information propagation.
        dry_run:
            Whether or not to apply the new state.
            Applies the new states and updates histories (and returns them).
            Returns a state_history, action_history, landing_state tuple.
            Also applies them to the environment, unless dry_run=True.
        seed:
            If a seed is specified, builds a new random state.
            (e.g. to perform different simulations with different seeds).
            If None, uses the environment's internal random state.
        """

        # Get random state:
        rs = self.random if (seed is None) else np.random.RandomState(seed)
        
        step_count = 0
        state_history = []
        action_history = []
        current_state = self.state.copy()
        for _ in range(n_steps):
            
            # Get action from policy:
            if self.policy:
                action = self.policy.get_action(current_state)
            else:
                action = Action(env=self, selected_ids=[])  # Placeholder action.
            
            # Simulate organic transmission and apply intervention:
            probs = TransitionMatrix.agent_probabilities(env=self, state=current_state, action=action)

            # Simulate an outcome for each agent:
            landing_state = rs.binomial(n=1, p=probs).astype(bool)
            landing_state = State(env=self, vector=landing_state)
            
            # Store results:
            step_count += 1
            state_history.append(current_state)
            action_history.append(action)
            current_state = landing_state

        # Update history and apply new state:
        if not dry_run:
            self.step_count += step_count
            self.state_history.extend(state_history)
            self.action_history.extend(action_history)
            self.update_state(new_state=landing_state)

        return state_history, action_history, landing_state

    def build_sub_problem(self, agents, new_env_params=None):
        """
        Instantiate a subproblem involving a sub-group of agents
        (i.e. ignore inner and outer circle connections to any other agents).
        Returns a new environment with a modified copy of each agent.

        agents:
            List of agent objects for the sub-problem.
        new_env_params:
            (Optional) A dictionary of parameters for the new Environment.
            By default, the new Environment will be initialized with the same
            parameters as this one. Any values in the new parameter dict
            will override the current ones.
        """
        
        # Get agent_ids that are part of the sub-problem:
        sub_agent_ids = [agent.id for agent in agents]
        sub_workplace_ids = sorted( set().union(*[agent.workplace_ids for agent in agents]) )
        sub_specialty_ids = sorted( set().union(*[agent.specialty_ids for agent in agents]) )
        def filter_list(vals, sub_vals):
            if vals is None:
                return None
            return [val for val in vals if val in sub_vals]
        # Build copies of agents without connections outside the group:
        sub_agents = []
        for agent in agents:
            # Build copy of agent:
            sub_agent = Agent(
                env = None,
                workplace_ids = filter_list(vals=agent.workplace_ids, sub_vals=sub_workplace_ids),
                specialty_ids = filter_list(vals=agent.specialty_ids, sub_vals=sub_specialty_ids),
                inner_circle = filter_list(vals=agent.inner_circle, sub_vals=sub_agent_ids),
                outer_circle = filter_list(vals=agent.outer_circle, sub_vals=sub_agent_ids),
                informed_init = agent.informed_init,
                receptivity = agent.receptivity,
                persuasiveness = agent.persuasiveness,
            )
            sub_agent.id = agent.id
            sub_agent.informed = agent.informed
            sub_agent.intervention = agent.intervention
            # Add copied agent to list:
            sub_agents.append(sub_agent)

        # Create new environment with for the subproblem:
        env_params = {
            'agents' : sub_agents,
            'agent_ids' : sub_agent_ids,
            'seed' : self.random.randint(1e8),  # Build a new seed.
            'base_receptivity' : self.base_receptivity,
            'base_persuasiveness' : self.base_persuasiveness,
            'intervention_size' : self.intervention_size,
            'influence_model' : self.influence_model,
            'transition_model' : self.transition_model,
            'policy_model' : self.policy_model,
        }
        if new_env_params is not None:
            assert 'agents' not in new_env_params, "Should not override the list of (sub) agents for the new Environment."
            assert 'agents_ids' not in new_env_params, "Should not override the list of (sub) agents for the new Environment."
            env_params.update(new_env_params)
        env = Environment(**env_params)
        return env

    def get_workplace(self, workplace_id):
        """
        Returns a list of agents in a given workplace.
        """
        return [agent for agent in self.agents.values() if workplace_id in agent.workplace_ids]

    def get_speciality(self, speciality_id):
        """
        Returns a list of agents in a given speciality.
        """
        return [agent for agent in self.agents.values() if speciality_id in agent.speciality_ids]

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
            #raise RuntimeError("self.build_transition_matrix has not been called.")
        return self.trans.T

    @property
    def landing_states(self):
        if self.trans is None:
            return None
            #raise RuntimeError("self.build_transition_matrix has not been called.")
        return self.trans.landing_states

    @property
    def starting_states(self):
        if self.trans is None:
            return None
            #raise RuntimeError("self.build_transition_matrix has not been called.")
        return self.trans.starting_states

    @property
    def state_space(self):
        if self.trans is None:
            return None
            #raise RuntimeError("self.build_transition_matrix has not been called.")
        return self.trans.state_space

    @property
    def action_space(self):
        if self.trans is None:
            return None
            #raise RuntimeError("self.build_transition_matrix has not been called.")
        return self.trans.action_space

    @property
    def n_informed(self):
        return self.state.n_informed

    def __str__(self):
        return f"<Environment with {self.state.n_informed}/{self.state.n_agents} informed agents>"
