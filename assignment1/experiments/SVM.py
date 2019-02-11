import warnings

import numpy as np
import sklearn

import experiments
import learners


class SVMExperiment(experiments.BaseExperiment):
    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose

    def perform(self):
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/SVM.py
        samples = self._details.ds.features.shape[0]
        features = self._details.ds.features.shape[1]

        gamma_fracs = np.arange(1/features, 2.1, 0.2)
        tols = np.arange(1e-8, 1e-1, 0.01)
        C_values = np.arange(0.001, 2.5, 0.25)
        iters = [-1, int((1e6/samples)/.8)+1]

        best_params_linear = None
        best_params_rbf = None
        # Uncomment to select known best params from grid search. This will skip the grid search and just rebuild
        # the various graphs
        #
        # Dataset 1 (credit default):
        '''
        best_params_linear = {
            "C": 0.251,
            "class_weight": "balanced",
            "dual": False,
            "fit_intercept": True,
            "intercept_scaling": 1,
            "loss": "squared_hinge",
            "max_iter": 42,
            "multi_class": "ovr",
            "penalty": "l2",
            "tol": 0.020000010000000002,
            "verbose": False
          }
        best_params_rbf = {
            "C": 0.751,
            "cache_size": 200,
            "class_weight": "balanced",
            "coef0": 0,
            "decision_function_shape": "ovo",
            "degree": 3,
            "gamma": 0.043478260869565216,
            "kernel": "rbf",
            "max_iter": -1,
            "probability": False,
            "shrinking": True,
            "tol": 0.08000001,
            "verbose": False
          }
        # Dataset 2:
        best_params_linear = {
            "C": 0.251,
            "class_weight": "balanced",
            "dual": False,
            "fit_intercept": True,
            "intercept_scaling": 1,
            "loss": "squared_hinge",
            "max_iter": 42,
            "multi_class": "ovr",
            "penalty": "l2",
            "tol": 0.020000010000000002,
            "verbose": False
          }
        best_params_rbf = {
            "C": 1.501,
            "cache_size": 200,
            "class_weight": "balanced",
            "coef0": 0,
            "decision_function_shape": "ovo",
            "degree": 3,
            "gamma": 0.0056179775280898875,
            "kernel": "rbf",
            "max_iter": -1,
            "probability": False,
            "shrinking": True,
            "tol": 0.09000000999999999,
            "verbose": False
          }
        '''
        # Linear SVM
        params = {'SVM__max_iter': iters, 'SVM__tol': tols, 'SVM__class_weight': ['balanced'],
                  'SVM__C': C_values}
        complexity_param = {'name': 'SVM__C', 'display_name': 'Penalty', 'values': np.arange(0.001, 2.5, 0.1)}

        iteration_details = {
            'x_scale': 'log',
            'params': {'SVM__max_iter': [2**x for x in range(12)]},
        }

        # NOTE: If this is causing issues, try the RBFSVMLearner. Passing use_linear=True will use a linear kernel
        #       and passing use_linear=False will use the RBF kernel. This method is slower but if libsvm is not
        #       available it may be your only option
        learner = learners.LinearSVMLearner(dual=False)
        if best_params_linear is not None:
            learner.set_params(**best_params_linear)

        best_params = experiments.perform_experiment(
            self._details.ds, self._details.ds_name, self._details.ds_readable_name, learner, 'SVMLinear', 'SVM',
            params, complexity_param=complexity_param, seed=self._details.seed, iteration_details=iteration_details,
            best_params=best_params_linear,
            threads=self._details.threads, verbose=self._verbose)

        of_params = best_params.copy()
        learner = learners.LinearSVMLearner(dual=True)
        if best_params_linear is not None:
            learner.set_params(**best_params_linear)
        experiments.perform_experiment(self._details.ds, self._details.ds_name, self._details.ds_readable_name, learner,
                                       'SVMLinear_OF', 'SVM', of_params, seed=self._details.seed,
                                       iteration_details=iteration_details,
                                       best_params=best_params_linear,
                                       threads=self._details.threads, verbose=self._verbose,
                                       iteration_lc_only=True)

        # RBF SVM
        params = {'SVM__max_iter': iters, 'SVM__tol': tols, 'SVM__class_weight': ['balanced'],
                  'SVM__C': C_values,
                  'SVM__decision_function_shape': ['ovo', 'ovr'], 'SVM__gamma': gamma_fracs}
        complexity_param = {'name': 'SVM__C', 'display_name': 'Penalty', 'values': np.arange(0.001, 2.5, 0.1)}

        learner = learners.SVMLearner(kernel='rbf')
        if best_params_rbf is not None:
            learner.set_params(**best_params_rbf)
        best_params = experiments.perform_experiment(
            self._details.ds, self._details.ds_name, self._details.ds_readable_name, learner, 'SVM_RBF', 'SVM',
            params, complexity_param=complexity_param, seed=self._details.seed, iteration_details=iteration_details,
            best_params=best_params_rbf,
            threads=self._details.threads, verbose=self._verbose)

        of_params = best_params.copy()
        learner = learners.SVMLearner(kernel='rbf')
        if best_params_rbf is not None:
            learner.set_params(**best_params_rbf)
        experiments.perform_experiment(self._details.ds, self._details.ds_name, self._details.ds_readable_name, learner,
                                       'SVM_RBF_OF', 'SVM', of_params, seed=self._details.seed,
                                       iteration_details=iteration_details,
                                       best_params=best_params_rbf,
                                       threads=self._details.threads, verbose=self._verbose,
                                       iteration_lc_only=True)
