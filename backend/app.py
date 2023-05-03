from textblob import TextBlob
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import csv
import os
import json
from flask import Flask, render_template, request
from flask_cors import CORS
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import io
from flask import Response

pand = "python -m pip install pandas"
skl = "python -m pip install scikit-learn"
nump = "python pip install numpy"
tb = "python -m pip install textblob"
os.system(pand)
os.system(skl)
os.system(nump)
os.system(tb)


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




    # Top 5 features between mov descript and book descript
    feature_names = vectorizer.get_feature_names()
    movie_feature_scores = list(zip(feature_names, movie_vector.toarray()[0]))
    top_features = sorted(movie_feature_scores, key=lambda x: x[1], reverse=True)[:5]
    # Top 5 features between mov rev and book descp
    movie_feature_scores_sec = list(zip(feature_names, movie_rev_vector.toarray()[0]))
    top_features_new = sorted(movie_feature_scores_sec, key=lambda x: x[1], reverse=True)[:5]


    # combine features
    features = top_features + top_features_new
    print("Top 5 feature of movie descript + review ", features)

    # get tfidf vectors for books and extract scores for each feature
    book_scores = {}
    for book in top_books:
        description = book_description[book]
        book_vector = vectorizer.transform([description])
        book_scores[book] = [book_vector.toarray()[0][vectorizer.vocabulary_[feat]] for feat, score in features]


    plot(book_scores, movie_title)
    # # plot movie description and review vector scores
    # ax.plot(features, movie_desc_vector.toarray()[0][vectorizer.vocabulary_[feat]] for feat in movie_desc_features)
    # ax.plot(features, movie_review_vector.toarray()[0][vectorizer.vocabulary_[feat]] for feat in movie_review_features)


    out = []
    for i, book in enumerate(top_books):
        out.append(book)
    return out


@ app.route("/")
def home():
    return render_template('base.html', title="sample html")

@app.route('/plot')
def plot(book_scores, movie_title):
    fig, ax = plt.subplots(figsize=(5,5)) # Adjust the size of the figure
    for book, scores in book_scores.items():
        ax.plot(scores, label=book)

    ax.set_xlabel('Features')
    ax.set_ylabel('TF-IDF Score')
    ax.set_title(f'TF-IDF Scores for {movie_title} and related books')
    
    # Adjust the font size of the legend
    ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    
    # Save the plot to a memory buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight') # Use bbox_inches='tight' to avoid cutting off the legend
    buffer.seek(0)

    filepath = os.path.join('static', 'images', 'plot.png')
    with open(filepath, 'wb') as f:
        f.write(buffer.getvalue())

    # Send the image as a response
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
    print(data)
    return json.dumps(data)


if __name__ == '__main__':
    app.run(debug=True)
