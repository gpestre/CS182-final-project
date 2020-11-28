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
            If not provided, defaults to sorted order of agent_ids.
        """

        # Bind simulation environment:
        self.env = env

        # Set whether or not this matrix encodes inner or outer circle of relationships:
        self.valid_scopes = {'inner','outer','both'}
        self.scope = scope
        assert scope in self.valid_scopes, f"{scope} is not a valid scope: {self.valid_scopes}"

        # Maintain a list of agent_ids in the order the appear in the matrix:
        self.agent_ids = agent_ids if agent_ids else list(sorted(env.agents.keys()))

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

    def __init__(self, env, model='default', agent_ids=None):
        """
        env:
            The simulation environment.
        method:
            Hyperparameter to control which model is used.
        agent_ids:
            (Optional) List of agent_ids in the order they should be stored in the matrix.
            If not provided, defaults to sorted order of agent_ids.
        """
        self.env = env
        # Set influence model:
        self.valid_models = {'default'}
        self.model = model
        assert model in self.valid_models, f"{model} is not a valid model: {self.valid_models}"

        # Maintain a list of agent_ids in the order the appear in the matrix:
        self.agent_ids = agent_ids if agent_ids else list(sorted(env.agents.keys()))

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


class State:

    """
    Static representation of a state of the environment
    (i.e. whether or not each professional is informed).
    """

    def __init__(self, env, informed=None):
        """
        env:
            The simulation environment.
        informed:
            A list of agent_ids who are informed or a dict of booleans keyed by agent_id.
            If not provided, constructed from current environment state.
            Providing `informed` as a dict is faster but does not check key validity.
        """
        self.env = env
        self.informed = {agent_id:False for agent_id in env.agents.keys()}
        if informed is None:
            for agent_id, agent in env.agents.items():
                self.informed[agent_id] = agent.informed
        elif isinstance(informed, dict):
            self.informed = informed  # Directly insert dict without checking it.
        elif isinstance(informed, list):
            for agent_id in informed:
                self.informed[agent_id] = True
        elif isinstance(informed, State):
            for agent_id, val in informed.informed.items():
                self.informed[agent_id] = val
        else:
            raise ValueError("informed should be list or dict.")
        self.n_agents = len(self.env.agent_ids)

    @property
    def vector(self):
        """
        Return the state as a numpy boolean array in the same order as the adjacency matrix.
        """
        informed = {agent_id for agent_id,val in self.informed.items() if val}
        vector = np.array([(agent_id in informed) for agent_id in self.env.agent_ids])
        return vector

    @property
    def n_informed(self):
        return sum(self.informed.values())

    def __str__(self):
        return f"<State with {self.n_informed}/{self.n_agents} informed agents>"
    
    def copy(self):
        return State(self.env, self.informed)


class Action:

    """
    Static representation of an action in the environment
    (i.e. whether or not to intervene for each individual).
    """

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


    def __init__(self, env, selected, value=None):
        """
        env:
            The simulation environment.
        selected:
            A list of agent_ids selected for intevention or a dict of booleans keyed by agent_id.
            Providing `selected` as a dict is faster but does not check key validity.
        """
        self.env = env
        self.selected = {agent_id:False for agent_id in env.agents.keys()}
        if isinstance(selected, dict):
            self.selected = selected  # Directly insert dict without checking it.
        elif isinstance(selected, list):
            for agent_id in selected:
                self.selected[agent_id] = True
        else:
            raise ValueError("selected should be list or dict.")
        self.n_agents = len(self.env.agent_ids)
        # Optionally assign estimated value of this intervention:
        self.value = value

    @property
    def vector(self):
        """
        Return the action as a numpy boolean array in the same order as the adjacency matrix.
        """
        selected = {agent_id for agent_id,val in self.selected.items() if val}
        vector = np.array([(agent_id in selected) for agent_id in self.env.agent_ids])
        return vector

    @property
    def n_selected(self):
        return sum(self.selected.values())

    def __str__(self):
        return f"<Action with {self.n_selected}/{self.n_agents} selected agents>"
    
    def copy(self):
        return Action(self.env, self.selected)

    def get_value(self, state=None, method='random'):
        if self.value:  # Return a manually set value if it exists.
            return self.value
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

    def __init__(self, agents, agent_ids=None, seed=None):
        """
        agents:
            A list or dict (keyed by agent_id) of Agent objects.
        agent_ids:
            (Optional) List of agent_ids in the order they should be stored in the matrix.
            If not provided, defaults to sorted order of agent_ids.
        """

        # Hyperparameters:
        self.base_receptivity = 0.5
        self.base_persuasiveness = 0.5
        self.intervention_size = 100  # How many interventions at each step?

        # Random state:
        self.seed = seed
        self.random = np.random.RandomState(seed)

        # Network structure:
        self.agents = Agent.to_dict(agents)
        self.workplaces = dict()
        self.specialties = dict()
        
        # Maintain a list of agent_ids in the order the appear in the matrix:
        self.agent_ids = agent_ids if agent_ids else list(sorted(self.agents.keys()))

        # Network state (set by update function):
        self.state = None  # Boolean indicators of which agents are informed.
        self.inner = None  # AdjacnecyMatrix for inner circle networks.
        self.outer = None  # AdjacnecyMatrix for outer circle networks.

        # Networks Graph
        self.G = None
        self.pos = None
        self.edge_labels = None

        # Transition Matrix
        self.T = None
        self.action_space = None
        self.state_space = None

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
        self.update_network_graph()

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

    def update_influence(self):
        """
        Rebuild agent adjacency matrix.
        """
        self.influence = InfluenceMatrix(self, model='default', agent_ids=self.agent_ids)

    def update_state(self, informed=None):
        """
        Build state object to reflect current agent states.
        informed:
            A list of agent_ids who are informed or a dict of booleans keyed by agent_id.
            (Optional -- if not provided, agent states are left unchanged.)
        """

        # Apply specified state, if provided (otherwise keep current state):
        if informed:
            # Build a state object (which accepts `informed` in various formats):
            new_state = State(self, informed)
            # Update agents in the environment to reflect requested state:
            for agent_id, val in zip(self.agents, new_state.vector):
                self.agents[agent_id].informed = val

        # Update state (i.e. which agents are informed):
        self.state = State(self)

    def update_transition_matrix(self, intervention_n=2):
        """
        WARNING - Only run for small problem sizes

        intervention_n:
            Number of agents being informed at each timestep
        """
        self.action_space = list(itertools.combinations(self.agent_ids, intervention_n))
        self.state_space = list(itertools.product([0,1], repeat=len(self.agent_ids)))

        self.T = np.zeros((len(self.action_space), len(self.state_space), len(self.state_space)))

        for i, actions in enumerate(self.action_space):
            for j, state1 in enumerate(self.state_space):
                for k, state2 in enumerate(self.state_space):

                    # Check that the new state is consistent with the action alone
                    consistent = 0
                    for action in actions:
                        if state2[action] == 1:
                            consistent += 1
                        
                    if consistent == intervention_n:
                    
                        # Calculate the probabilities of influence occuring to each next state
                        total_influence_prob = []
                        consistency_check = 1

                        for n_state, n_val in enumerate(state2):
                            next_state_prob = []
                            if n_state in actions:
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
                                            next_state_prob.append(1 - self.influence.matrix[c_state, n_state])
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

    def update_network_graph(self, iterations=250):
        """

        """
        # Initialize the graph
        self.G = nx.DiGraph()

        # Create the Graph structure
        for agent in self.agents.values():
            for next_agent_id in agent.inner_circle:
                connection_strength = self.influence.matrix[(agent.id, next_agent_id)]
                self.G.add_edge(agent.id, next_agent_id, val=connection_strength)

            for next_agent in agent.outer_circle:
                connection_strength = self.influence.matrix[(agent.id, next_agent_id)]
                self.G.add_edge(agent.id, next_agent, val=connection_strength)

        self.pos = nx.spring_layout(self.G, iterations=iterations)
        self.edge_labels = dict([((node1, node2, ), f'{connection_data["val"]}\n\n{self.G.edges[(node2,node1)]["val"]}')
                for node1, node2, connection_data in self.G.edges(data=True) if self.pos[node1][0] > self.pos[node2][0]])

    def plot_network_graph(self):
        """

        """
        if self.G is None:
            self.update_network_graph()

        _, ax = plt.subplots()
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=self.edge_labels, font_color='red')
        nx.draw_networkx(self.G, self.pos, with_labels=True, node_size=400, ax=ax, connectionstyle='arc3, rad = 0.1')

    @property
    def n_informed(self):
        return self.state.n_informed

    def __str__(self):
        return f"<Environment with {self.state.n_informed}/{self.state.n_agents} informed agents>"
