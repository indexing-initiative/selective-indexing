"""
Script to run ensemble
and save validation and test predictions to file.
"""

import json
import pandas as pd

from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from item_select import ItemSelector


def run_model(X, y, model, dataset):
    """
    Load and run trained model
    on the validation and test datasets.
    Save the respective files.
    """

    y_probs = model.predict_proba(X)[:, 0]
    df_results = pd.DataFrame(data = {
            'y_true': y,
            'y_prob': y_probs,
            'pmid': X['pmid'],
            'journal': X['journals']
            })
    df_results.to_csv("../results/{}_voting_predictions.csv".format(dataset), sep=",", index=None)


def preprocess_json_data(citations):
    """
    Preprocess data for ensemble.
    Return dictionary of lists of each feature.
    """

    pmids, titles, abstracts, affiliations, journal_nlmid, labels = [], [], [], [], [], []

    for citation in citations:
        pmids.append(citation['pmid'])
        if citation['title'] == "":
            titles.append("None")
        else:
            titles.append(citation['title'])
        if citation['abstract'] == "":
            abstracts.append("None")
        else:
            abstracts.append(citation['abstract'])
        if citation['author_list'] == "":
            affiliations.append("None")
        else:
            affiliations.append(citation['author_list'])

        journal_nlmid.append(citation['journal_nlmid'])
        labels.append(citation['is_indexed'])

    citations = {
        'abstract': abstracts,
        'titles': titles,
        'author_list': affiliations,
        'journals': journal_nlmid,
        'pmid': pmids}

    return citations, labels


def parse_citations(XML_path):
    """
    Parse the validation and test
    data from json.
    """

    with open(XML_path, encoding="utf8") as f:
        citations_json = json.load(f)

    return citations_json


def main():
    """
    Load trained ensemble, and run on
    validation and test data.
    """

    print("Making selective indexing predictions")
    model = joblib.load("voting_model.joblib")
    datasets = ["validation", "test"]
    for dataset in datasets:
        if dataset == "validation":
            XML_path = "../datasets/pipeline_validation_set.json"
        else:
            XML_path = "../datasets/pipeline_test_set.json"

        citations = parse_citations(XML_path)
        X, y = preprocess_json_data(citations)

        run_model(X, y, model, dataset)


if __name__ == '__main__':
    main()
