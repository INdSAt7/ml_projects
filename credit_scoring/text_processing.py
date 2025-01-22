import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def preprocess_text_dataframe(df, text_columns, max_features=5000, fitted_vectorizers=None):
    """
    Preprocess and convert multiple text columns in a DataFrame into numerical features using TF-IDF.
    Ensures that vectorized features have consistent size across datasets.

    :param df: Input DataFrame
    :param text_columns: List of text column names to process
    :param max_features: Maximum number of features for TF-IDF vectorization
    :param fitted_vectorizers: Dictionary of pre-fitted TfidfVectorizer objects (optional)
    :return: Dictionary of transformed TF-IDF features and fitted TF-IDF vectorizers for each column
    """
    def preprocess_text(text):
        """
        Preprocess individual text by converting to lowercase, removing punctuation,
        and handling non-string entries.
        """
        if not isinstance(text, str):
            return ''  # Convert non-string entries to empty strings
        text = text.lower()  # Convert to lowercase
        text = ''.join([char for char in text if char.isalnum() or char.isspace()])  # Remove punctuation
        return text

    tfidf_results = {}
    if fitted_vectorizers is None:
        fitted_vectorizers = {}

    for text_column in text_columns:
        # Preprocess text data
        df[text_column + '_cleaned'] = df[text_column].apply(preprocess_text)

        # Initialize or use a pre-fitted TF-IDF vectorizer
        if text_column in fitted_vectorizers:
            vectorizer = fitted_vectorizers[text_column]
            X_text = vectorizer.transform(df[text_column + '_cleaned'])
        else:
            vectorizer = TfidfVectorizer(max_features=max_features)
            X_text = vectorizer.fit_transform(df[text_column + '_cleaned'])
            fitted_vectorizers[text_column] = vectorizer

        # Store results in a dictionary
        tfidf_results[text_column] = (X_text, vectorizer)

    return tfidf_results, fitted_vectorizers
