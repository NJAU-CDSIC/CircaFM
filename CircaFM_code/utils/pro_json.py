import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    auc
)
from typing import Tuple
import json


def calculate_binary_metrics(json_path: str) -> Tuple[float, float, float, float, float, float]:
    """
    Compute six core metrics for binary classification.
    Expects a JSON with keys 'targets', 'preds', 'scores'.
    Returns: accuracy, precision, recall, f1, auroc, aupr
    """
    with open(json_path, 'r') as file:
        test_res = json.load(file)

    y_true = np.array(test_res['targets'])
    y_pred = np.array(test_res['preds'])
    y_score = np.array(test_res['scores'])[:, 1]   # probability of positive class

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Non-binary labels detected. Use calculate_multiclass_metrics instead.")

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)

    return acc, pre, rec, f1, auroc, aupr


def calculate_multiclass_metrics(json_path: str, verbose: bool = True) -> dict:
    """
    Compute multi-label classification metrics.
    Supports two JSON formats:
      1. Dict format: target/preds/scores are dicts keyed by class name.
      2. 2D list format: targets/preds/scores are lists of lists (samples × classes).
    Returns a dict containing overall and class-wise metrics.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Convert 2D list to dict if necessary
    if 'targets' in data and isinstance(data['targets'], list) and len(data['targets']) > 0:
        class_names = ['period', 'phase', 'amplitude', 'baseline']
        targets_np = np.array(data['targets'])
        preds_np   = np.array(data['preds'])
        scores_np  = np.array(data['scores'])

        data['target'] = {cn: targets_np[:, i].tolist() for i, cn in enumerate(class_names)}
        data['preds']  = {cn: preds_np[:, i].tolist() for i, cn in enumerate(class_names)}
        data['scores'] = {cn: scores_np[:, i].tolist() for i, cn in enumerate(class_names)}

    test_res = data

    metrics_dict = {
        'overall': {},
        'class_wise': {}
    }
    class_names = list(test_res['target'].keys())

    def aggregate_data(key):
        return np.concatenate([test_res[key][cn] for cn in class_names])

    # Overall micro-average
    targets_all = aggregate_data('target')
    preds_all   = aggregate_data('preds')
    scores_all  = aggregate_data('scores')

    metrics_dict['overall']['report'] = classification_report(
        targets_all, preds_all,
        target_names=['0', '1'],
        output_dict=True,
        zero_division=0
    )

    if len(np.unique(targets_all)) < 2:
        metrics_dict['overall']['roc_auc'] = np.nan
        metrics_dict['overall']['aupr'] = np.nan
        if verbose:
            print("Warning: overall targets contain only one class, AUC set to NaN.")
    else:
        precision, recall, _ = precision_recall_curve(targets_all, scores_all)
        metrics_dict['overall']['roc_auc'] = roc_auc_score(targets_all, scores_all)
        metrics_dict['overall']['aupr'] = auc(recall, precision)

    # Class-wise metrics
    for class_name in class_names:
        target = np.array(test_res['target'][class_name])
        pred   = np.array(test_res['preds'][class_name])
        score  = np.array(test_res['scores'][class_name])

        metrics_dict['class_wise'][class_name] = {
            'report': classification_report(
                target, pred,
                target_names=['0', '1'],
                output_dict=True,
                zero_division=0
            ),
        }

        if len(np.unique(target)) < 2:
            metrics_dict['class_wise'][class_name]['roc_auc'] = np.nan
            metrics_dict['class_wise'][class_name]['aupr'] = np.nan
            if verbose:
                print(f"Warning: class '{class_name}' has only one unique label, AUC set to NaN.")
        else:
            precision, recall, _ = precision_recall_curve(target, score)
            metrics_dict['class_wise'][class_name]['roc_auc'] = roc_auc_score(target, score)
            metrics_dict['class_wise'][class_name]['aupr'] = auc(recall, precision)

    if verbose:
        _pretty_print(metrics_dict)

    return metrics_dict


def _pretty_print(metrics_dict: dict):
    """Print metrics in a formatted table."""
    print("\n" + "=" * 60)
    print("OVERALL METRICS (micro-average)")
    print("=" * 60)
    print(f"ROC-AUC: {metrics_dict['overall']['roc_auc']:.4f}")
    print(f"AUPR:    {metrics_dict['overall']['aupr']:.4f}")
    print("\nClassification Report (overall):")
    print(classification_report_from_dict(metrics_dict['overall']['report']))

    print("\n" + "=" * 60)
    print("CLASS-WISE METRICS")
    print("=" * 60)
    for cls_name, cls_metrics in metrics_dict['class_wise'].items():
        print(f"\n--- {cls_name} ---")
        print(f"ROC-AUC: {cls_metrics['roc_auc']:.4f}")
        print(f"AUPR:    {cls_metrics['aupr']:.4f}")
        print("Classification Report:")
        print(classification_report_from_dict(cls_metrics['report']))


def classification_report_from_dict(report_dict: dict) -> str:
    """Convert a classification report dict to a readable string."""
    report_copy = {k: v for k, v in report_dict.items() if k not in ('accuracy',)}
    df = pd.DataFrame(report_copy).transpose()
    return df.to_string(float_format="%.4f")