import re
import string
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator


class CleanText(BaseEstimator, TransformerMixin):

    def process_text(self, X, y=None):

        text = X.lower()  # makes lower case
        text = re.sub(r'[\w-]+@([\w-]+\.)+[\w-]+',
                      '', text)  # remove words with @
        text = re.sub(r'[%s]' % re.escape(string.punctuation),
                      '', text)  # removes punctuation
        text = re.sub(r'\w*\d\w*', '', text)  # removes words with numbers
        # removes carriage returns line breas tabs replace wuith space
        text = re.sub('/(\r\n)+|\r+|\n+|\t+/i', ' ', text)
        return text

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        # return self.spacy(X)
        return X.apply(self.process_text)


def build_model():
    """This function builds a new model and returns it.

    The model should be implemented as a sklearn Pipeline object.

    Your pipeline needs to have two steps:
    - preprocessor: a Transformer object that can transform a dataset
    - model: a predictive model object that can be trained and generate predictions

    :return: a new instance of your model
    """

    text_pipeline = Pipeline(
        [("clean", CleanText()), ("tfidf", TfidfVectorizer(stop_words='english', max_df=0.5, ngram_range=(1, 2)))])
    preprocessor = ColumnTransformer([("text_pipe", text_pipeline, "text")])
    return Pipeline([("preprocessor", preprocessor), ("model", MultinomialNB(alpha=0.01))])
