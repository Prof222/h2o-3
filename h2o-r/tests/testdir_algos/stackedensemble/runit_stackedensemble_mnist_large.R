setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")

stackedensemble.mnist.test <- function() {

    train_file <- locate("bigdata/laptop/mnist/train.csv.gz")
    test_file <- locate("bigdata/laptop/mnist/test.csv.gz")
    train <- h2o.importFile(train_file)
    test <- h2o.importFile(test_file)
    y <- "C785"
    x <- setdiff(names(train), y)
    train[,y] <- as.factor(train[,y])
    test[,y] <- as.factor(test[,y])
    # Number of CV folds (to generate level-one data for stacking)
    nfolds <- 3

    # 1. Generate a 2-model ensemble (GBM + RF)
    # Train & Cross-validate a GBM
    my_gbm <- h2o.gbm(x = x,
                    y = y,
                    training_frame = train,
                    nfolds = nfolds,
                    ntrees = 10,
                    fold_assignment = "Modulo",
                    keep_cross_validation_predictions = TRUE,
                    seed = 1)

    # Train & Cross-validate a RF
    my_rf <- h2o.randomForest(x = x,
                    y = y,
                    training_frame = train,
                    nfolds = nfolds,
                    ntrees = 10,
                    fold_assignment = "Modulo",
                    keep_cross_validation_predictions = TRUE,
                    seed = 1)

    # Train a stacked ensemble using the GBM and RF above
    ensemble <- h2o.stackedEnsemble(y = y,
                training_frame = train,
                base_models = list(my_gbm, my_rf))

    # Eval ensemble performance on a test set
    perf <- h2o.performance(ensemble, newdata = test)

    # Compare to base learner performance on the test set
    perf_gbm_test <- h2o.performance(my_gbm, newdata = test)
    perf_rf_test <- h2o.performance(my_rf, newdata = test)
    baselearner_best_test <- min(h2o.mean_per_class_error(perf_gbm_test), h2o.mean_per_class_error(perf_rf_test))
    ensemble_test <- h2o.mean_per_class_error(perf)
    print(sprintf("Best Base-learner Test Mean Per class Error:  %s", baselearner_best_test))
    print(sprintf("Ensemble Test Mean Per class Error:  %s", ensemble_test))
    expect_equal(TRUE,ensemble_test <= baselearner_best_test)

}

doTest("Stacked Ensemble MNIST Test", stackedensemble.mnist.test)
