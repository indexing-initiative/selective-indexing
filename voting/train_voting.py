"""
Script to train the voting classifier used
for selective indexing system
"""

from multiprocessing import cpu_count, Pool
import pandas as pd
import numpy as np

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from parse_citations import CitationParser
from item_select import ItemSelector


def get_pipeline(model):
    """
    Return the pipeline
    """

    pipeline = Pipeline([
        ('union', FeatureUnion(
            transformer_list=[
                ('titles', Pipeline([
                    ('selector', ItemSelector(column='titles')),
                    ('tfidf', TfidfVectorizer()),
                ])),
                ('author_pipe', Pipeline([
                    ('selector', ItemSelector(column='author_list')),
                    ('tfidf', TfidfVectorizer()),
                ])),
                ('text_pipe', Pipeline([
                    ('selector', ItemSelector(column='abstract')),
                    ('tfidf', TfidfVectorizer()),
                ])),
                ],
            )),
            ('ensemble', model),
        ])

    return pipeline


def filter_citations(df):
    """
    The dataframe contains citations still in xml format.
    Parse it out with CitationParser.
    """

    filtered_df = df.apply(CitationParser.filter_citations, axis=1)
    filtered_df.dropna(inplace=True)
    filtered_df.reset_index(inplace=True, drop=True)
    # Get abstracts with the citation parser. Returns dataframe
    citations_df = filtered_df.apply(CitationParser.get_abstract, axis=1)
    # Get affiliations
    affiliation_list = filtered_df.apply(CitationParser.get_affiliations, axis=1)
    citations_df['author_list'] = affiliation_list
    # Get titles
    titles = filtered_df.apply(CitationParser.get_title, axis=1)
    citations_df['titles'] = titles

    return citations_df


def model_vote(training_data):
    """
    Train the ensemble, and save
    it to joblib.
    """

    print("Training model optimized for fbeta, where beta=2...")
    models = [
        ('sgd', SGDClassifier(loss='modified_huber', alpha=.0001, max_iter=1000)),
        ('lg', LogisticRegression(C=2, random_state=0)),
        ('bnb', BernoulliNB(alpha=.01)),
        ('rfc', RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0))
        ]

    voting_model = VotingClassifier(estimators=models, voting='soft', n_jobs=8)
    pipeline = get_pipeline(voting_model)
    pipeline.fit(training_data, training_data['label'])
    print("Saving model...")
    joblib.dump(pipeline, "voting_model.joblib")


def get_data():
    """
    Load the data from json into pandas
    dataframe and beginning training.
    """

    print("Getting data")
    training_data = "training_citations_2017.json"
    df_train = pd.read_json(training_data, encoding="utf-8", orient="split")

    print("Training data shape:", df_train.shape)

    partitions = 6
    df_split = np.array_split(df_train, partitions)
    pool = Pool(partitions)
    df_processed_train = pd.concat(pool.map(filter_citations, df_split))
    pool.close()
    pool.join()
    df_processed_train.reset_index(inplace=True, drop=True)

    model_vote(df_processed_train)


if __name__ == '__main__':
    get_data()
