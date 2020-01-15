import json
import joblib
import plotly
import pandas as pd
import numpy as np

from flask import Flask, render_template, request
from plotly.graph_objs import Bar, Heatmap

import sqlite3

# NLP import
import spacy
nlp = spacy.load("en_core_web_sm")


app = Flask(__name__)


def tokenize(text):
    """ Return lowercase lemmatized tokens of the input text.
        Remove punctuation, spacy.stop_words, and pronouns.

    Args:
        text (str): input text
    Returns:
        tokens ([str]): list of lower case lemmatized tokens
    """

    doc = nlp(text)
    tokens = [tok.lemma_.lower() for tok in doc
              if not (tok.is_stop or tok.is_punct or (tok.lemma_ == '-PRON-'))]
    return tokens


# load data
conn = sqlite3.connect('./data/DisasterResponse.db')
df = pd.read_sql_query('SELECT * FROM message', conn, index_col='id')

# load model
model = joblib.load("./models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    #################################
    #################################

    categories = df.loc[:, 'request':]\
                   .assign(uncategorized=np.where(df.related == 0, 1, 0))

    # genre distribution
    ####################
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # categories distribution
    #########################
    cat_counts = categories.sum().sort_values(ascending=False)
    # extract category names for plot
    cat_counts_cat_names = [x.replace('_', ' ').title()
                            for x in cat_counts.index.values]

    # Source(genre) distribution per category
    #########################################
    source_sum = df.genre.to_frame().merge(categories, on='id')\
                   .groupby('genre').agg('sum').T
    # generate percent of each genre for each category
    for col in source_sum:
        source_sum[col + 'pct'] = (100.0 * source_sum[col] /
                                   (source_sum.direct + source_sum.news +
                                    source_sum.social))
    # reorder categories by number of messages in decreasing order,
    # same as cat_counts
    source_sum = source_sum.reindex(cat_counts.index.to_list())
    # extract category names for plot
    source_sum_cat_names = [x.replace('_', ' ').title()
                            for x in source_sum.index.values]

    # correlation between categories
    ################################
    corr = categories.corr()
    # extract category names for plot
    corr_cat_names = [x.replace('_', ' ').title() for x in corr.index.values]

    # create visuals
    #################################

    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # Genre Distribution
        {
            'data': [
                Bar(
                    x=[x.title() for x in genre_names],
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Sources',
                'xaxis': {
                    'title': "Source"
                },
                'yaxis': {
                    'title': "Count"
                },
                'width': 1000,
                'margin': dict(
                    l=200,
                    r=30,
                    b=180,
                    t=30,
                    pad=4
                ),
            }
        },

        # Categories Distribution
        {
            'data': [
                Bar(
                    y=cat_counts_cat_names,
                    x=cat_counts.values,
                    orientation="h"
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'xaxis': {
                    'title': "Count",
                    'automargin': "true",
                },
                'yaxis': {
                    'autorange': "reversed"
                },
                'height':800,
                'width': 1000,
                'margin': dict(
                    l=200,
                    r=30,
                    b=180,
                    t=30,
                    pad=4
                )
            }
        },

        # Percent distribution of messages by sources(genre) for each cat
        {
            'data': [
                Bar(
                    name='Direct',
                    x=source_sum.directpct.values,
                    y=source_sum_cat_names,
                    orientation='h'
                ),
                Bar(
                    name='News',
                    x=source_sum.newspct.values,
                    y=source_sum_cat_names,
                    orientation='h'
                ),
                Bar(
                    name='Social Media',
                    x=source_sum.socialpct.values,
                    y=source_sum_cat_names,
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Sources of Disaster Messages By Category',
                'xaxis': {
                    'title': "Percent of Messages From Source (%)"
                },
                'yaxis': {
                    'autorange': "reversed"
                },
                'barmode': "stack",
                'height':800,
                'margin': dict(
                    l=200,
                    r=30,
                    b=180,
                    t=30,
                    pad=4
                ),
            }
        },

        # Categories Correlation
        {
            'data': [
                Heatmap(
                    z=corr.to_numpy(),
                    x=corr_cat_names,
                    y=corr_cat_names,
                    colorscale='Blues'
                )
            ],

            'layout': {
                'title': 'Correlation Between Message Categories',
                'yaxis': {
                    'autorange': "reversed"
                },
                'height':800,
                'width': 1000,
                'margin': dict(
                    l=200,
                    r=30,
                    b=180,
                    t=30,
                    pad=4
                ),
            },
        },


    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == '__main__':
    main()
