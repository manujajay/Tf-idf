# TF-IDF

This repository explores Term Frequency-Inverse Document Frequency (TF-IDF), a statistical measure used to evaluate how important a word is to a document in a collection or corpus. It is widely used in text mining and information retrieval.

## Prerequisites

- Python 3.6 or higher
- Libraries: NumPy, Scikit-learn

## Installation

To install the necessary Python libraries:

```bash
pip install numpy scikit-learn
```

## Example - Calculating TF-IDF

This example demonstrates how to calculate the TF-IDF scores for a set of documents using Scikit-learn's `TfidfVectorizer`.

### `calculate_tfidf.py`

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "TF-IDF is a statistical measure that evaluates how relevant a word is to a document in a collection of documents.",
    "It is often used as a weighting factor in information retrieval and text mining.",
    "This approach to TF-IDF calculation is typical in search engine scoring and ranking of a document's relevance given a user query."
]

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Print the TF-IDF values
feature_names = vectorizer.get_feature_names_out()
for doc_num, doc in enumerate(tfidf_matrix):
    print(f"Document {doc_num + 1}")
    for col in doc.nonzero()[1]:
        print(f"{feature_names[col]}: {doc[0, col]:.3f}")
```

## Contributing

If you are interested in contributing to the development of TF-IDF related projects, feel free to fork this repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
