# The Doctor (doesn't?) Know Best

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#experiments">Experiments</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>

  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

In this project, we borrow from the extensive literature on influence maximization to model the flow of information within the medical community. We then consider the following problem: given an effective, but costly information intervention, what is the optimal manner in which to "seed" this intervention throughout the population? More concretely, supposing a benevolent social planner could invite only a strict subset of doctors to a conference in which they would learn the current standard of care, what would be the optimal subset of doctors to invite?

We specify the problem as a MDP and show that the full problem is only computationally tractable for trivially small networks. We then turn to approximate solutions such as traditional network interrogation techniques such as degree centrality and a custom implemented hierarchical MDP approach.

<!-- Getting Started -->
## Getting Started

### Prerequisites

Prior to running the code in this project an installation of both Python 3 and a package manager is required, the instructions for installation in this guide will be utilizing the conda package manager but the equivalent process can be performed in any of your choosing. 

### Installation

1. Clone the github repository
    ```sh
    git clone git@github.com:gpestre/CS182-final-project.git
    ```
2. Create and activate a virtual environment
    ```sh
    conda env create -f environment.yml
    conda activate cs182
    ```
3. (Optional) If you do not have access to an IDE which can run Jupyter notebooks you can install Jupyter into the conda environment.
    ```sh
    conda install jupyter
    ```


### Experiments

This repository comes preloaded with the three experiments which we discussed in the paper. These experiments can be found within the "notebook/" folder labelled as 1, 2, and 3. To run these experiments please initialize a Jupyter environment, either through an IDE such as VSCode or directly in the command line using:
```sh
jupyter notebook
```

These premade experiments are designed to replicate the results found in the report but as also useful examples for how instantiate and create your own network and policy solver utilizing the backend classes. The three experiments cover:

1. Baseline small scale network
2. Larged scale disparate network
3. Approximation of the Cambridge, MA medical environment

## Usage

* To do, add further details for how to manually interface with the classes and data structures in the back end.