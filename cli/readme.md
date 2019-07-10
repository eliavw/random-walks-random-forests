# How to use this CLI

This directory contains some CLI to run sets of experiments.

## Examples

### 1. Most basic case

For a local execution of all the settings in the `config/<folder-name>` folder.  
Run command:
    `./cli.py -c <folder-name> -l`

- `c` flag indicates a config file **or** folder. This folder is expected to be found _inside_ the `config/` folder.
- `l` flag indicates we want to execute *locally*
