# model_performance_metrics
This module robustly calculate the most useful model performance metrics implemented in
scikit-learn, AUC_mu and gAUC for binary and multi-class models. 
The util is model-agnostic in the sense that one can will calculate
the metrics as long as the prediction labels and probabilities are
provided, along with the "true" values from a validation sample and training data.

An implementation of the gAUC metric is also provided.
