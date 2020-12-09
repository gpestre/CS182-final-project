import os

import numpy as np
import pandas as pd

from structures import *

class Dataset:

    def __init__(self, seed=182):

        assert len(Agent.all_agents)==0, "Please call Agent.reset() to restart agent_ids at zero before building a dataset."

        self.random = np.random.RandomState(seed)

        self.agents = []
        self.workplace_ids = []
        self.specialty_ids = []

        self.env = None

    def build_environment(self, *args, **kwargs):

        self.env = Environment(
            agents = self.agents,
            *args, **kwargs
        )

        return self.env


    def build_workplace(self,
        workplace_id,
        n_agents,
        specialty_proportions,
        inner_circle_func,
        outer_circle_func,
        informed_func,
        receptivity_func,
        persuasiveness_func,
    ):
        
        agent_id_start = len(Agent.all_agents)  # This is the ID that will be assigned to the next new agent.
        agent_ids = list( np.arange(agent_id_start,agent_id_start+n_agents) )
        
        specialty_proportions = list( np.array(specialty_proportions)/sum(specialty_proportions) )
        specialty_ids = list(range(1,1+len(specialty_proportions)))
        specialty_id_pool = list( self.random.choice(specialty_ids, p=specialty_proportions, size=n_agents) )

        new_agents = []
        for agent_id, specialty_id in zip(agent_ids, specialty_id_pool):
            agent = Agent(
                workplace_ids = [workplace_id],
                specialty_ids = [specialty_id],
                inner_circle = inner_circle_func(agent_ids),
                outer_circle = outer_circle_func(agent_ids),
                informed_init = informed_func(),
                receptivity = np.round(receptivity_func(),2),
                persuasiveness = np.round(persuasiveness_func(),2),
            )
            assert agent.id == agent_id, f"Something has gone wrong 😱 (sometimes happens when using autoreload in a Jupyter notebook)."
            new_agents.append(agent)

        # Update global lists:
        self.agents.extend(new_agents)
        for specialty_id in specialty_ids:
            if specialty_id not in set(self.specialty_ids):
                self.specialty_ids.append(specialty_id)
        if workplace_id not in set(self.workplace_ids):
            self.workplace_ids.append(workplace_id)

    def connect_workplaces(self,
        workplace_id_1,
        workplace_id_2,
        p_speciality,
        p_other,
        max_connections,
    ):

        assert workplace_id_1 in self.workplace_ids, f"workplace_id={workplace_id_1} does not exist."
        assert workplace_id_2 in self.workplace_ids, f"workplace_id={workplace_id_2} does not exist."

        agents_1 = [agent for agent in self.agents if (workplace_id_1 in agent.workplace_ids)]
        agents_2 = [agent for agent in self.agents if (workplace_id_2 in agent.workplace_ids)]

        max_connections = float("+Inf") if max_connections is None else max_connections

        n_connections = 0
        for agent1 in agents_1:
            for agent2 in agents_2:
                if n_connections >= max_connections:
                    break
                if agent2.id == agent1.id:
                    continue
                if agent2.id in agent1.inner_circle:
                    continue
                if agent2.id in agent1.outer_circle:
                    continue
                if agent1.specialty_ids[0]==agent2.specialty_ids[0]:
                    connect = self.random.uniform()<p_speciality
                else:
                    connect = self.random.uniform()<p_other
                if connect:
                    n_connections += 1
                    agent1.outer_circle.append(agent2.id)

    def build_agent_table(self):

        table = []
        for agent in self.agents:
            agent_info = {
                'agent_id' : agent.id,
                'workplace_id' : agent.workplace_ids[0],
                'specialty_id' : agent.specialty_ids[0],
                'inner_circle' : ";".join([str(x) for x in agent.inner_circle]),
                'inner_circle_size' : len(agent.inner_circle),
                'outer_circle' : ";".join([str(x) for x in agent.outer_circle]),
                'outer_circle_size' : len(agent.outer_circle),
                'informed_init' : agent.informed_init,
                'receptivity' : agent.receptivity,
                'persuasiveness' : agent.persuasiveness,
            }
            table.append(agent_info)
        table = pd.DataFrame(table)
        return table

    def recipe1(self):
        
        workplace_sizes = [10,20,5,4,4]
        specialty_proportions = [10,3,4,2]
        
        # Build workplaces"
        for w,n_agents in enumerate(workplace_sizes):
            self.build_workplace(
                workplace_id=w+1,
                n_agents=n_agents,
                inner_circle_func = lambda agent_ids: self.rand_choice_func(n_min=1, n_max=4, method='uniform')(agent_ids),
                outer_circle_func = lambda agent_ids: self.rand_choice_func(n_min=3, n_max=7, method='uniform')(agent_ids),
                specialty_proportions=specialty_proportions,
                informed_func = lambda: self.random.binomial(n=1, p=0.1),
                receptivity_func = lambda: self.random.uniform(0.0, 0.5),
                persuasiveness_func = lambda: self.random.uniform(0.0, 0.5),
            )
        
        # Build connections between workplaces:
        for _ in range(3):
            assert len(self.workplace_ids) >= 2, "Need at least 2 workplaces to connect."
            workplace_id_1, workplace_id_2 = self.random.choice(self.workplace_ids, size=2)
            self.connect_workplaces(
                workplace_id_1 = workplace_id_1,
                workplace_id_2 = workplace_id_2,
                p_speciality = 0.10,
                p_other = 0.05,
                max_connections = 5,
            )

        self.build_environment()

    def rand_choice_func(self, n_min, n_max, method='uniform'):
            if method=='uniform':
                def func(vals):
                    n_choices = self.random.randint(n_min, n_max)
                    n_choices = min(n_choices, len(vals))
                    return list(np.random.choice(vals, size=n_choices, replace=False))
                return func
            else:
                raise NotImplementedError

    def save_csv(self, filepath, overwrite=False):

        if os.path.isfile(filepath) and (not overwrite):
            raise FileExistsError(f"{filepath} exists and overwrite=False.")

        df = self.build_agent_table()
        df.to_csv(filepath, index=False)
        print("Saved {filepath} .")

    @classmethod
    def load_csv(cls, filepath):

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"{filepath} does not exist.")
        
        # Build dataset object:
        dataset = cls(seed=None)  # There is no practical way to reapply the same seed.
        
        # Load table:
        table = pd.read_csv(filepath)
        for i,row in table.iterrows():
            # Build agent:
            agent = Agent(
                workplace_ids = [row['workplace_id']],
                specialty_ids = [row['specialty_id']],
                inner_circle = [int(x) for x in row['inner_circle'].split(";")],
                outer_circle = [int(x) for x in row['outer_circle'].split(";")],
                informed_init = row['informed_init'],
                receptivity = row['receptivity'],
                persuasiveness = row['persuasiveness'],
            )
            assert agent.id == row['agent_id'], "Numbering should match if Agent.reset() was called before building."
            # Add agent to Dataset container:
            dataset.agents.append(agent)
        # Add workplaces and specialities to Dataset container:
        dataset.workplace_ids = sorted(set(table['workplace_id']))
        dataset.specialty_ids = sorted(set(table['specialty_id']))

        return dataset


if __name__=="__main__":

    filepath = "../outputs/dataset1.csv"

    # Build test:
    Agent.reset()  # Reset IDs to zer.
    ds = Dataset(seed=182)
    ds.recipe1()
    ds.save_csv(filepath, overwrite=False)

    # Load test:
    Agent.reset()  # Reset IDs to zero.
    env = Dataset.load_csv(filepath).build_environment(seed=123)  # Note: Environment seed and Dataset seed are unrelated.
    print(env)
