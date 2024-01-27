import sklearn.metrics as metrics
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from model_metrics.gauc import gauc
from model_metrics.auc_mu import auc_mu  # find it here: https://github.com/kleimanr/auc_mu


def all_model_metrics(
    y_true: np.array,
    y_pred_labels: np.array,
    y_pred_probabilities: np.array,
    y_train: np.array,
    target_type: str,
    zero_division: int = 0,
):
    """
    Calculate all model performance metrics in sk-learn - and a few more
    like e.g. AUC_mu and gAUC for binary and multi-class models.

    The util is model-agnostic in the sense that one can will calculate
    the metrics as long as the prediction labels and probabilities are
    provided, along with the "true" values from a validation sample.

    Parameters
    ----------
    y_true
        True values of the samples
    y_pred_labels
        Model predictions with labels
    y_pred_probabilities
        Model prediction probabilities
    y_train
        Training data of the model
    target_type
        "binary" or "multiclass"
    zero_division:
        Give classes where zero-division occurs a "bad" assessment

    Returns
    -------
    dict
        All relevant model performance metrics
    """
    # one-hot encode the prediction using labels from training data
    label_binarizer = LabelBinarizer().fit(y_train)
    y_val_onehot = label_binarizer.transform(y_true)

    # add try/except flow to some functions
    def roc_auc_score(y_true, y_score, average, multi_class):
        try:
            return metrics.roc_auc_score(
                y_true=y_true, y_score=y_score, average=average, multi_class=multi_class
            )
        except ValueError:
            return np.nan

    # get all
    model_performance_metrics = {
        "accuracy": metrics.accuracy_score(y_true=y_true, y_pred=y_pred_labels),
        "balanced accuracy score": metrics.balanced_accuracy_score(
            y_true=y_true, y_pred=y_pred_labels
        ),
        "confusion matrix": metrics.confusion_matrix(
            y_true=y_true, y_pred=y_pred_labels
        ).tolist(),
        "log loss": metrics.log_loss(y_true=y_val_onehot, y_pred=y_pred_probabilities),
        "matthews corrcoef": metrics.matthews_corrcoef(
            y_true=y_true, y_pred=y_pred_labels
        ),
        "ndcg score": metrics.ndcg_score(
            y_true=y_val_onehot, y_score=y_pred_probabilities
        ),
    }

    # multiclass only
    if target_type == "multiclass":
        fpr_ovr_micro, tpr_ovr_micro, _ = metrics.roc_curve(
            y_val_onehot.ravel(), y_pred_probabilities.ravel()
        )
        dict_metric_values_mc = {
            "gAUC": gauc(y_true, y_pred_labels)["statistic"],
            "AUC_mu": auc_mu(
                y_true=y_true, y_score=y_pred_probabilities, stop_on_errors=False
            ),
            "AUC_ovo_macro": roc_auc_score(
                y_true=y_true,
                y_score=y_pred_probabilities,
                average="macro",
                multi_class="ovo",
            ),
            "AUC_ovo_weighted": roc_auc_score(
                y_true=y_true,
                y_score=y_pred_probabilities,
                average="weighted",
                multi_class="ovo",
            ),
            "AUC_ovr_macro": roc_auc_score(
                y_true=y_true,
                y_score=y_pred_probabilities,
                average="macro",
                multi_class="ovr",
            ),
            "AUC_ovr_micro": metrics.auc(fpr_ovr_micro, tpr_ovr_micro),
            "AUC_ovr_weighted": roc_auc_score(
                y_true=y_true,
                y_score=y_pred_probabilities,
                average="weighted",
                multi_class="ovr",
            ),
            "average precision score_macro": metrics.average_precision_score(
                y_true=y_val_onehot, y_score=y_pred_probabilities, average="macro"
            ),
            "average precision score_micro": metrics.average_precision_score(
                y_true=y_val_onehot, y_score=y_pred_probabilities, average="micro"
            ),
            "average precision score_weighted": metrics.average_precision_score(
                y_true=y_val_onehot, y_score=y_pred_probabilities, average="weighted"
            ),
            "F1 score_macro": metrics.f1_score(
                y_true=y_true,
                y_pred=y_pred_labels,
                average="macro",
                zero_division=zero_division,
            ),
            "F1 score_micro": metrics.f1_score(
                y_true=y_true,
                y_pred=y_pred_labels,
                average="micro",
                zero_division=zero_division,
            ),
            "F1 score_weighted": metrics.f1_score(
                y_true=y_true,
                y_pred=y_pred_labels,
                average="weighted",
                zero_division=zero_division,
            ),
            "precision recall F-score support_macro": metrics.precision_recall_fscore_support(
                y_true=y_true,
                y_pred=y_pred_labels,
                average="macro",
                zero_division=zero_division,
            ),
            "precision recall F-score support_micro": metrics.precision_recall_fscore_support(
                y_true=y_true,
                y_pred=y_pred_labels,
                average="micro",
                zero_division=zero_division,
            ),
            "precision recall F-score support_weighted": metrics.precision_recall_fscore_support(
                y_true=y_true,
                y_pred=y_pred_labels,
                average="weighted",
                zero_division=zero_division,
            ),
        }
        model_performance_metrics.update(dict_metric_values_mc)
    # binary only
    elif target_type == "binary":
        dict_metric_values_bin = {
            "AUC": roc_auc_score(y_true=y_true, y_score=y_pred_probabilities),
            "average precision score": metrics.average_precision_score(
                y_true=y_true, y_score=y_pred_labels
            ),
            "brier score loss": metrics.brier_score_loss(
                y_true=y_true, y_prob=y_pred_probabilities
            ),
            "F1 score": metrics.f1_score(
                y_true=y_true,
                y_pred=y_pred_labels,
                average="binary",
                zero_division=zero_division,
            ),
            "precision recall curve": metrics.precision_recall_curve(
                y_true=y_true, probas_pred=y_pred_probabilities
            ),
            "precision recall F-score support": metrics.precision_recall_fscore_support(
                y_true=y_true,
                y_pred=y_pred_labels,
                average="binary",
                zero_division=zero_division,
            ),
        }
        model_performance_metrics.update(dict_metric_values_bin)
    else:
        raise ValueError("Invalid target type.")

    return model_performance_metrics
