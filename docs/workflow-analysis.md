# Workflow Analysis

## 2. Analysis

### 2.1 Collection of results

This happens on the **remote**

Go to `note/collect-results` and run the notebook `collect-results-parallel`.
This one works both in the case of sequential or parallel experiments.

What it does is that in the folder `prod/RunExp/<expid>/results`, it will create
four files that summarize the
    
    1. Model configurations
    2. Qry codes
    3. Results
    4. Timings
    
### 2.2 Transfer

We need to do the transfer remote->local.

1. Copy files from `prod/RunExp/<expid>` to your local machine

This folder contains all the collected results, which you made happen in the
previous step.
    
### 2.3 Visualisation

Go to `note/visuals` and run the `visual-sandbox` notebook for the basic webapp,
which allows some interesting views of the results.

