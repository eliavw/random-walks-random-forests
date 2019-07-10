# Workflow Remote

This document describes how to run a script remotely.

## 1. Config and Run

### 1.1 Generate configuration

Every experiment is run based on a configuration file. In `note/config-generation`
you find the notebooks that can be used to generate those files.

    1. Generate a json file using such a notebook
    2. Copy the json file (or its folder, in case of multiple files)
    to the cmd/config folder ON THE SERVER

### 1.2 Generate queries

This is optional, but for completeness. MERCS is a bit special in that it is
multi-directional, which complicates matters, especially in evaluation. 

From a practical point of view, what this means is that every experiment 
typically consists of a bunch of tiny prediction tasks, which we will commonly
refer to as `queries` because they ought to simulate a (random) user using a
trained MERCS model for all kinds of stuff.

These queries are a sub-part of the master configuration of an experiment, but
since they need to be somewhat persistent, we generate them in a seperate process.

To do so,

    1. Generate a query-config and query-codes using the notebook to be found in
    under note/query-generation
    2. This notebook has sufficiently functionality to output everything in the right
    location, locally.
    3. For execution on the server, copy the `resc/query` folder to the remote 
    server.
   

### 1.3 Execution

For a remote execution of all the settings in the `ijcai-remote` folder.  The 
`nodefile` file in the `cmd` folder deserves some special attention. This is
essentially a config script used by gnu-parallel (the command line package we use
for the very simple parallel execution) that indicates which machines it can use,
and how many processes it is allowed to start on each machine.
    
    1. Check and edit `nodefile`
    2. ssh to a master machine on the remote cluster.
    3. `go homework/cli`
        - Navigates to correct folder
        - Opens screen
        - Sets python path
    4. `./cli.py -c ijcai-remote`

N.B.:
- `go homework/cli` is managed in `.bashrc`
- `c` flag indicates a config file **or** folder 
- Do not put the `l` flag that indicates local running.

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


