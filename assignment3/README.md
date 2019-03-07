# Unsupervised Learning and Dimensionality Reduction

## Initial setup
Edit `./experiments/__init__.py` and change `BEST_NN_PARAMS` to match the best NN architecture for one of your data sets (if at some point in the future you need to do this for both data sets this will need up be updated).

1. Run the various experiments (perhaps via `python run_experiment.py --all`)
2. Update the dim values in `run_clustering.sh` based on the optimal values found in 2 (perhaps by looking at the scree graphs)
3. Run `run_clustering.sh`


## Output
Output CSVs and images are written to `./output` and `./output/images` respectively. Sub-folders will be created for each DR algorithm (ICA, PCA, etc) as well as the benchmark.

If these folders do not exist the experiments module will attempt to create them.

## Clustering Experiments

The experiments will output modified versions of the data sets after applying the DR methods. The script `run_clustering.sh` can be used to perform clustering on these modified datasets, using a specific number of components for the DR method.

**BE SURE TO UPDATE THE VALUES IN THIS SCRIPT FOR YOUR DATASETS**. 

There are different optimal values for each algorithm and each dataset, and using the wrong value will make you a sad panda.


## Graphing

The run_experiment script can be use to generate plots via:

```
python run_experiment.py --plot
```

Since the files output from the experiments follow a common naming scheme this will determine the problem, algorithm,
and parameters as needed and write the output to sub-folders in `./output/images`.

