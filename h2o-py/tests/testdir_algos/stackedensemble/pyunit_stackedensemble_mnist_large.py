#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import print_function

import h2o

import sys
sys.path.insert(1,"../../../")  # allow us to run this standalone

from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from tests import pyunit_utils


def stackedensemble_mnist_test():

    train = h2o.import_file(path=pyunit_utils.locate("bigdata/laptop/mnist/train.csv.gz"))
    test = h2o.import_file(path=pyunit_utils.locate("bigdata/laptop/mnist/test.csv.gz"))

    y = "C785"
    x = list(range(784))
    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()
    # Number of CV folds (to generate level-one data for stacking)
    nfolds = 3

    # Train and cross-validate a GBM
    my_gbm = H2OGradientBoostingEstimator(distribution="multinomial",
                                          nfolds=nfolds,
                                          ntrees=10,
                                          fold_assignment="Modulo",
                                          keep_cross_validation_predictions=True,
                                          seed=1)
    my_gbm.train(x=x, y=y, training_frame=train)
    perf_gbm_test = my_gbm.model_performance(test_data=test)

    # Train and cross-validate a RF
    my_rf = H2ORandomForestEstimator(nfolds=nfolds,
                                     ntrees=10,
                                     fold_assignment="Modulo",
                                     keep_cross_validation_predictions=True,
                                     seed=1)

    my_rf.train(x=x, y=y, training_frame=train)
    perf_rf_test = my_rf.model_performance(test_data=test)

    # Train a stacked ensemble using the GBM and GLM above
    stack = H2OStackedEnsembleEstimator(model_id="my_ensemble_mnist",
                                        base_models=[my_gbm.model_id,  my_rf.model_id])

    stack.train(x=x, y=y, training_frame=train)
    perf_stack_test = stack.model_performance(test_data=test)

    baselearner_best_mean_per_class_error_test = min(perf_gbm_test.mean_per_class_error(), perf_rf_test.mean_per_class_error())
    stack_mean_per_class_error_test = perf_stack_test.mean_per_class_error()
    print("Best Base-learner Test Mean Per class Error:  %s", baselearner_best_mean_per_class_error_test)
    print("Ensemble Test Mean Per class Error:  %s", stack_mean_per_class_error_test)
    assert stack_mean_per_class_error_test <= baselearner_best_mean_per_class_error_test

if __name__ == "__main__":
    pyunit_utils.standalone_test(stackedensemble_mnist_test)
else:
    stackedensemble_mnist_test()


