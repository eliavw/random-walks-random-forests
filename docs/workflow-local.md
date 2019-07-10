# Workflow Local

This document describes how to run a script locally.

## 1. Config and Run

### 1.1 Generate configuration

Every experiment is run based on a configuration file. In `note/config-generation`
you find the notebooks that can be used to generate those files.

    1. Generate a json file using such a notebook
    2. Copy the json file (or its folder, in case of multiple files)
    to the cmd/config folder.

### 1.2 Generate queries

Only needs to be done if you need new queries.

MERCS is a bit special in that it is multi-directional, which complicates matters, especially in evaluation. 

From a practical point of view, what this means is that every experiment 
typically consists of a bunch of tiny prediction tasks, which we will commonly
refer to as `queries` because they ought to simulate a (random) user using a
trained MERCS model for a *prediction task that was not known at training time*.
Such a prediction task is what we call a query.

These queries can be regarded as part of the configuration of an experiment, but
since they can perfectly be re-used across experiments, we generate them in a seperate process.

To do so,

    1. Generate a query-config and query-codes using the notebook to be found in
    under note/query-generation
    2. This notebook has sufficiently functionality to output everything in the right
    location, so there is no step 2.
    
Be careful though, you can override old queries and if that happens, you cannot
compare to older experiments.

### 1.3 Execution

For a local execution of all the settings in the `ijcai-local` folder.  
Run command:
    `./cli.py -c ijcai-local -l`

- `c` flag indicates a config file **or** folder 
- `l` flag indicates we want to execute *locally*

## 2. Analysis and Visualisation

### 2.1 Collection of results

Go to `note/collect-results` and run the notebook `collect-results-parallel`.
This one works both in the case of sequential or parallel experiments.

What it does is that in the folder `prod/RunExp/<expid>/results`, it will create
four files that summarize the
    
    1. Model configurations
    2. Qry codes
    3. Results
    4. Timings
    
All of that stuff is then ready for further processing and visualisation.

### 2.2 Visualisation

Go to `note/visuals` and run the `visual-sandbox` notebook for the basic webapp,
which allows some interesting views of the results.


