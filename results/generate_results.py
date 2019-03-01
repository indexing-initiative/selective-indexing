"""
Generate results presented in selective indexing paper.
Three figures are generated, as well as a text file containing
precision and recall scores for ensemble, cnn, and combined models
at a series of thresholds.
"""


import json
import numpy as np
import sys

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, precision_recall_curve


def adjusted_classes(y_probs, cnn=False, combined=None, cnn_thresh=None, voting_thresh=None):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """

    if cnn:
        p = cnn_thresh
        return ["MEDLINE" if y >= p else "PubMed-not-MEDLINE" for y in y_probs]
    elif combined is not None:
        #print("Combined probs")
        p = combined
        return ["MEDLINE" if y >= p else "PubMed-not-MEDLINE" for y in y_probs]
    else:
        p = voting_thresh
        return ["MEDLINE" if y >= p else "PubMed-not-MEDLINE" for y in y_probs]


def zoomed_figure(test_voting_predictions, test_cnn_predictions, vxc):
    """
    Plot the models performance on one graph, zoomed in to just the
    high recall values.

    The initial threshold values for each model were chosen via trial
    and error, in order to get the lines for each model to cover
    a similar range of recall values.

    This function also generates the text file with precision and
    recall scores for each model at each threshold. The order of
    results if ensemble, cnn, combined.
    """

    voting_color = '#6890F0'
    voting_marker = 's'
    cnn_color = '#F08030'
    cnn_marker = 'v'
    combo_color = '#C03028'
    combo_marker = 'o'

    precision_list = []
    recall_list = []
    lower_ci_list = []
    upper_ci_list = []
    fig, ax = plt.subplots(figsize = (7, 5))
    SMALL = 7
    MED = 12
    BIG = 14

    voting_thresh = .052
    cnn_thresh = .045
    combined_thresh = .0035
    recall_thresh = .998

    results = open("results.txt", "w")
    results.close()
    results = open("results.txt", "a+")

    results.write("\nEnsemble:\n")
    while True:
        preds_m = adjusted_classes(test_voting_predictions['y_prob'], False, voting_thresh=voting_thresh)
        precision = precision_score(test_voting_predictions['y_true'], preds_m, pos_label="MEDLINE")
        recall = recall_score(test_voting_predictions['y_true'], preds_m, pos_label="MEDLINE")
        precision_list.append(precision)
        recall_list.append(recall)
        results.write("{0}, {1}, {2}\n".format(precision, recall, voting_thresh))
        if recall > recall_thresh:
            break
        else:
            voting_thresh -= .0001
    print("Ensemble recall and precision and threshold:", precision, recall, voting_thresh, "\n")
    ax.plot(recall_list, precision_list, c=voting_color, marker=voting_marker, markevery=20, label="Ensemble")

    results.write("\nCNN:\n")
    precision_list = []
    recall_list = []
    lower_ci_list = []
    upper_ci_list = []
    while True:
        preds_a = adjusted_classes(test_cnn_predictions['y_prob'], True, cnn_thresh=cnn_thresh)
        precision = precision_score(test_cnn_predictions['y_true'], preds_a, pos_label="MEDLINE")
        recall = recall_score(test_cnn_predictions['y_true'], preds_a, pos_label="MEDLINE")
        #print(precision, recall, cnn_thresh)
        precision_list.append(precision)
        recall_list.append(recall)
        results.write("{0}, {1}, {2}\n".format(precision, recall, cnn_thresh))
        if recall > recall_thresh:
            break
        else:
            cnn_thresh -= .0001
    print("CNN recall and precision and threshold:", precision, recall, cnn_thresh, "\n")
    ax.plot(recall_list, precision_list, c=cnn_color, marker=cnn_marker, markevery=20, label="CNN")

    results.write("\nCombined:\n")
    precision_list = []
    recall_list = []
    threshold_list = []
    lower_ci_list = []
    upper_ci_list = []
    while True:
        adjusted_preds_combined = adjusted_classes(vxc, combined=combined_thresh)
        precision = precision_score(test_voting_predictions['y_true'], adjusted_preds_combined, pos_label="MEDLINE")
        recall = recall_score(test_voting_predictions['y_true'], adjusted_preds_combined, pos_label="MEDLINE")
        #print(precision, recall, combined_thresh)
        precision_list.append(precision)
        recall_list.append(recall)
        threshold_list.append(combined_thresh)
        results.write("{0}, {1}, {2}\n".format(precision, recall, combined_thresh))
        if recall > recall_thresh:
            break
        else:
            combined_thresh -= .0001
    print("Combined recall, precision and threshold:", precision, recall, combined_thresh, "\n")
    ax.plot(recall_list, precision_list, c=combo_color, marker=combo_marker, markevery=2, label="Combined model")

    results.close()

    ax.set_title('Precision-Recall', fontsize=BIG)
    ax.set_xlabel('Recall', fontsize=MED)
    ax.set_ylabel('Precision', fontsize=MED)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor = (1, .5), fontsize=MED)
    fig.savefig('all_models_precision_recall_zoom.png', bbox_inches = 'tight')
    plt.show()
    plt.close()


def plot_group_curve(combined_predictions):
    """
    Generate figure comparing the group performance of the
    combined model.
    """

    with open("group_ids.json", "r") as f:
        group_ids = json.load(f)

    groups = [
            'All journals',
            'chemistry',
            'science',
            'jurisprudence',
            'biotech'
            ]

    colors = [
            '#C03028',
            '#F08030',
            '#6890F0',
            '#78C850',
            '#A890F0',
            ]

    markers = [
            'o',
            'v',
            's',
            'P',
            'd'
            ]

    fig, ax = plt.subplots(figsize = (7, 5))
    SMALL = 7
    MED = 12
    BIG = 14

    for i, group in enumerate(groups):
        if group == "All journals":
            y_probs = combined_predictions['combined_probs']
            y_test = combined_predictions['y_true']
        else:
            predictions = combined_predictions.loc[combined_predictions['journal'].isin(group_ids[group])]
            y_probs = predictions['combined_probs']
            y_test = predictions['y_true']

        precision, recall, thresholds = precision_recall_curve(y_test, y_probs, pos_label='MEDLINE')
        if group == 'All journals':
            ax.plot(recall, precision, label=group, color=colors[i], marker=markers[i], markevery=1000, linewidth=1.0)
        elif group == 'jurisprudence':
            ax.plot(recall, precision, label="Jurisprudence", color=colors[i], marker=markers[i], markevery=300, linewidth=1.0)
        elif group == 'chemistry':
            ax.plot(recall, precision, label="Chemistry", color=colors[i], marker=markers[i], markevery=1000, linewidth=1.0)
        elif group == 'science':
            ax.plot(recall, precision, label="Science", color=colors[i], marker=markers[i], markevery=1000, linewidth=1.0)
        else:
            ax.plot(recall, precision, label="Biotech", color=colors[i], marker=markers[i], markevery=1000, linewidth=1.0)

    ax.set_title('Group Precision-Recall', fontsize=BIG)
    ax.set_xlabel('Recall', fontsize=MED)
    ax.set_ylabel('Precision', fontsize=MED)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor = (1, .5), fontsize = MED)
    fig.savefig('combined_model_groups.png', bbox_inches = 'tight')
    plt.show()
    plt.close()


def plot_all_models(voting_predictions, cnn_predictions, combined_predictions):
    """
    Plot the curve comparing the cnn and the
    ensemble to the combined model. Zoomed out version
    """

    models = [
            'Ensemble',
            'CNN',
            'Combined model'
            ]

    colors = [
            '#6890F0',
            '#F08030',
            '#C03028'
            ]

    markers = [
            's',
            'v',
            'o'
            ]

    fig, ax = plt.subplots(figsize = (7, 5))
    SMALL = 7
    MED = 12
    BIG = 14

    for i, df in enumerate([voting_predictions, cnn_predictions, combined_predictions]):
        if i != 2:
            y_probs = df['y_prob']
            y_test = df['y_true']
        else:
            y_probs = combined_predictions['combined_probs']
            y_test = combined_predictions['y_true']

        precision, recall, thresholds = precision_recall_curve(y_test, y_probs, pos_label='MEDLINE')
        ax.plot(recall, precision, label=models[i], c = colors[i], marker=markers[i], markevery=800, linewidth=1.0)

    ax.set_title('Precision-Recall', fontsize=BIG)
    ax.set_xlabel('Recall', fontsize=MED)
    ax.set_ylabel('Precision', fontsize=MED)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor = (1, .5), fontsize=MED)
    fig.savefig('all_models_precision_recall.png', bbox_inches = 'tight')
    plt.show()
    plt.close()


def main():
    """
    Get the data
    """

    print("Generating results...")
    # Could also use validation sets for these results:
    test_voting_predictions = pd.read_csv("test_voting_predictions.csv")
    test_cnn_predictions = pd.read_csv("test_cnn_predictions.csv", header=None, names=['pmid', 'pubdate', 'nlm_id', 'journal', 'group', 'y_true', 'y_prob'])
    test_voting_predictions['y_true'] = ["MEDLINE" if label == 1 else "PubMed-not-MEDLINE" for label in test_voting_predictions['y_true']]
    test_cnn_predictions['y_true'] = ["MEDLINE" if label == 1 else "PubMed-not-MEDLINE" for label in test_cnn_predictions['y_true']]
    test_cnn_predictions = test_cnn_predictions.sort_values(by=['pmid']).reset_index(drop=True)
    test_voting_predictions = test_voting_predictions.sort_values(by=['pmid']).reset_index(drop=True)

    # Check that the results from each model are compatible.
    for i, row in enumerate(test_cnn_predictions['pmid']):
        assert row == test_voting_predictions.loc[i, 'pmid'], print(row)
        assert test_cnn_predictions.loc[i, 'y_true'] == test_voting_predictions.loc[i, 'y_true']

    # Combine the probabilities
    combined_predictions = test_voting_predictions['y_prob']*test_cnn_predictions['y_prob']
    combined_df = pd.DataFrame({'combined_probs': combined_predictions, 'journal': test_voting_predictions['journal'], 'y_true': test_voting_predictions['y_true']})

    # Make zoomed in figure
    zoomed_figure(test_voting_predictions, test_cnn_predictions, combined_predictions)
    # Then make group per curve figure
    plot_group_curve(combined_df)
    # And finally the unzoomer precision recall curve
    plot_all_models(test_voting_predictions, test_cnn_predictions, combined_df)


if __name__ == "__main__":
    main()
