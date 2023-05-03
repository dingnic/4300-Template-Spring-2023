
import os

pand = "python -m pip install pandas"
skl = "python -m pip install scikit-learn"
nump = "python pip install numpy"
tb = "python -m pip install textblob"
mlib = "python -m pip install matplotlib"
os.system(pand)
os.system(skl)
os.system(nump)
os.system(tb)
os.system(mlib)

import matplotlib.pyplot as plt
import io
from flask import Response
from textblob import TextBlob
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import csv
import json
from flask import Flask, render_template, request
from flask_cors import CORS
from collections import defaultdict
import matplotlib


matplotlib.use('agg')
plt.rcParams['font.family'] = 'Courier New'


os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

MOVIES_CSV_PATH = os.path.join(
    os.environ['ROOT_PATH'], 'backend/rotten_tomatoes_movies.csv')

app = Flask(__name__)
CORS(app)

book_description = {}
movie_names = {}
movie_reviews = {}
movie_description = {}
movie_code_names = {}


def classify_review(review, positive_factor, neutral_factor, negative_factor):
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return positive_factor
    elif polarity < 0:
        return negative_factor
    else:
        return neutral_factor


# Keys are book names and values are list of reviews
book_review_sents = {}
positive_factor = 1.2
neutral_factor = 1.0
negative_factor = 0.8
with open('data/books_reviews.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        book = row['book_title']
        rev = row['review']
        rev_sent = classify_review(
            rev, positive_factor, neutral_factor, negative_factor)
        book_review_sents[book] = book_review_sents.get(book, []) + [rev_sent]


with open('data/books_description.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        book = row['book_title']
        desc = row['description']
        book_description[book] = desc
with open('data/movie_codes.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        movie_code = row['movie_code']
        movie_name = row['movie_name']
        movie_names[movie_code] = movie_name
        movie_code_names[movie_name] = movie_code
movie_name_codes = {v: k for k, v in movie_code_names.items()}

# keys are movie names and values are a list of the factors of the reviews
factor_dict = {}
with open('data/movie_reviews.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        movie_code = row['movie_code']
        rev = row['review']
        movie_reviews[movie_code] = rev
        # call function to get classification of the current review
        rev_sent = classify_review(
            rev, positive_factor, neutral_factor, negative_factor)
        factor_dict[movie_code] = factor_dict.get(movie_code, []) + [rev_sent]

with open('data/movie_description.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        movie_code = row['movie_code']
        desc = row['description']
        movie_description[movie_code] = desc

movie_list = list(movie_names.values())


def jaccard(movie, query):
    query = query.lower()
    movie = movie.lower()
    string1 = query.split()
    string2 = movie.split()
    set1 = set(string1)
    set2 = set(string2)

    if (movie in query):
        return 1
    jaccard_similarity = len(set1.intersection(set2)) / len(set1.union(set2))
    return jaccard_similarity


def cosine_sim_w_sent(movie_title, book_description, movie_reviews, movie_description, book_review_sents):
    movie_code = None
    for name, code in movie_code_names.items():
        if movie_title.lower() == name.lower():
            movie_code = code
            break
    if movie_code is None:
        print(f"No movie found with title '{movie_title}'")
        return
    movie_desc = movie_description[movie_code]
    book_desc = list(book_description.values())
    mov_revs = movie_reviews[movie_code]

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    book_vectors = vectorizer.fit_transform(book_desc)
    movie_vector = vectorizer.transform([movie_desc])
    movie_rev_vector = vectorizer.transform([mov_revs])

    similarity_scores = cosine_similarity(movie_vector, book_vectors)
    similarity_scores_sec = cosine_similarity(movie_rev_vector, book_vectors)

    alpha, beta = 0.5, 0.5
    # Calculate score with alpha, beta, and the sentiment_factor
    combined = alpha * similarity_scores + beta * \
        similarity_scores_sec

    combined_sent = np.zeros_like(combined)
    for i, val in enumerate(combined[0]):
        avg_sent = book_review_sents.get(
            list(book_description.keys())[i], [1])[0]
        combined_sent[0][i] = val*avg_sent

    top_indices = combined_sent.argsort()[0][-10:][::-1]
    top_books = [list(book_description.keys())[i] for i in top_indices]
    top_scores = {}
    dd = {}
    rd = {}
    for i, index in enumerate(top_indices):
        top_scores[top_books[i]] = combined_sent[0][index]
        dd[top_books[i]] = similarity_scores[0][index]
        rd[top_books[i]] = similarity_scores_sec[0][index]

    # Top 5 features between mov descript and book descript
    feature_names = vectorizer.get_feature_names()
    movie_feature_scores = list(zip(feature_names, movie_vector.toarray()[0]))
    top_features = sorted(movie_feature_scores,
                          key=lambda x: x[1], reverse=True)[:5]
    # Top 5 features between mov rev and book descp
    movie_feature_scores_sec = list(
        zip(feature_names, movie_rev_vector.toarray()[0]))
    top_features_new = sorted(movie_feature_scores_sec,
                              key=lambda x: x[1], reverse=True)[:5]

    # combine features
    features = top_features + top_features_new
    print("Top 5 feature of movie descript + review ", features)

    # get tfidf vectors for books and extract scores for each feature
    book_scores = {}
    for book in top_books:
        description = book_description[book]
        book_vector = vectorizer.transform([description])
        book_scores[book] = [book_vector.toarray()[0][vectorizer.vocabulary_[feat]]
                             for feat, score in features]

    plot(book_scores, movie_title, features,
         top_scores, dd, rd, book_review_sents)
    # # plot movie description and review vector scores
    # ax.plot(features, movie_desc_vector.toarray()[0][vectorizer.vocabulary_[feat]] for feat in movie_desc_features)
    # ax.plot(features, movie_review_vector.toarray()[0][vectorizer.vocabulary_[feat]] for feat in movie_review_features)

    out = []
    for i, book in enumerate(top_books):
        out.append(book)
    return out


def get_sent(num):
    num = num[0]
    if num == 1.2:
        return "+"
    if num == 1.0:
        return "0"
    if num == 0.8:
        return "-"
    else:
        return "0"


@ app.route("/")
def home():
    return render_template('base.html', title="sample html")


@app.route('/plot')
def plot(book_scores, movie_title, top_features, top_scores, dd, rd, book_review_sents):
    fig, ax = plt.subplots(figsize=(5, 5))  # Adjust the size of the figure
    fig.patch.set_facecolor('#808080')
    max_len = 0
    for i, tup in enumerate(list(book_scores.items())):
        book, scores = tup
        max_len = max(max_len, len(book))
    bound = 40
    if max_len > bound:
        max_len = bound + 3
    for i, tup in enumerate(list(book_scores.items())):
        book, scores = tup
        new_book = book
        if len(book) > bound:
            new_book = book[:bound + 1] + "..."
        line = ax.plot(
            scores, label=f"{new_book}" + " " * (max_len - len(new_book) + 2) + f"sim: {top_scores[book]:.3f}  d-d: {dd[book]:.3f}  d-r: {rd[book]:.3f}  sent: {get_sent(book_review_sents.get(book, [1]))}")

    features = [feat for feat, val in top_features]
    feature_vals = [val for feat, val in top_features]
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90)

    # plot movie
    #ax.plot(feature_vals, label=movie_title)
    ax.plot(feature_vals, label=movie_title,
            color='black', linestyle=':', linewidth=3)

    ax.set_xlabel('Features')
    ax.set_ylabel('TF-IDF Score')
    ax.set_title(f'TF-IDF Scores for {movie_title} and related books')

    legend = ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    # for text in legend.get_texts():
    #     text.set_fontname("Courier New")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)

    filepath = os.path.join('static', 'images', 'plot.png')
    with open(filepath, 'wb') as f:
        f.write(buffer.getvalue())

    return Response(buffer.getvalue(), mimetype='image/png')


@ app.route("/movie-search")
def movie_search():
    text = request.args.get("title")  # text = our input
    jaccard_similarities = np.array(
        [jaccard(text, movie) for movie in movie_list])

    # top 5 similar movies to the query (should be displayed to the user)
    top_indices = jaccard_similarities.argsort()[::-1][:5]
    sim_movies = []
    for i in top_indices:
        sim_movies.append(movie_list[i])

    return json.dumps(list(sim_movies))


@ app.route("/movies")
def episodes_search():
    movie = request.args.get("movie")
    print("The movie is: ", movie)
    # similarity_matrix, books_rev_ind, book_starting_index = logic(book_description, movie_code_names, movie_reviews)
    # print(factor_dict["Harry Potter and the Deathly Hallows - Part 2"])
    print("The code is: ", movie_code_names[movie])
    # print("The code is: ", movie_name_codes[movie])
    data = cosine_sim_w_sent(movie, book_description,
                             movie_reviews, movie_description, book_review_sents)
    return json.dumps(data)


if __name__ == '__main__':
    app.run(debug=True)
