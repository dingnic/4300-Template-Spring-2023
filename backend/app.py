import csv
import os
import json
from flask import Flask, render_template, request
from flask_cors import CORS
from collections import defaultdict
import random

# pand = "python -m pip install pandas"
# skl = "python -m pip install scikit-learn"
# nump = "python pip install numpy"
# os.system(pand)
# os.system(skl)
# os.system(nump)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

MOVIES_CSV_PATH = os.path.join(os.environ['ROOT_PATH'], 'backend/rotten_tomatoes_movies.csv')

app = Flask(__name__)
CORS(app)

book_description = defaultdict(str)
movie_names = defaultdict(str)
movie_reviews = defaultdict(str)
movie_description = defaultdict(str)
movie_code_names = defaultdict(str)

with open('data/books_description.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        book = row['book_title']
        desc = row['description']
        book_description[book] = desc
with open('data/movie_codes.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        movie_code= row['movie_code']
        movie_name = row['movie_name']
        movie_names[movie_code] = movie_name
        movie_code_names[movie_name] = movie_code
with open('data/movie_reviews.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        movie_code= row['movie_code']
        rev = row['review']
        movie_reviews[movie_code] = rev
with open('data/movie_description.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        movie_code= row['movie_code']
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

    if(movie in query):
        return 1
    jaccard_similarity = len(set1.intersection(set2)) / len(set1.union(set2))
    return jaccard_similarity


def cosine_sim(movie_title, book_description, movie_reviews, movie_description):
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

    vectorizer = TfidfVectorizer(max_features= 5000, stop_words='english')
    book_vectors = vectorizer.fit_transform(book_desc)
    movie_vector = vectorizer.transform([movie_desc])
    movie_rev_vector = vectorizer.transform([mov_revs])

    similarity_scores = cosine_similarity(movie_vector, book_vectors)
    similarity_scores_sec = cosine_similarity(movie_rev_vector, book_vectors)

    alpha, beta = 0.5, 0.5
    combined = alpha * similarity_scores + beta * similarity_scores_sec

    top_indices = combined.argsort()[0][-10:][::-1]
    top_books = [list(book_description.keys())[i] for i in top_indices]

    out = []
    for i, book in enumerate(top_books):
        out.append(book)
    return out
    




@app.route("/")
def home():
    return render_template('base.html', title="sample html")

@app.route("/movie-search")
def movie_search():
    text = request.args.get("title") #text = our input
    jaccard_similarities = np.array(
    [jaccard(text, movie) for movie in movie_list])

    # top 5 similar movies to the query (should be displayed to the user)
    top_indices = jaccard_similarities.argsort()[::-1][:5]
    sim_movies = []
    for i in top_indices:
        sim_movies.append(movie_list[i])

    return json.dumps(list(sim_movies))


@app.route("/movies")
def episodes_search():
    movie = request.args.get("movie")
    print(movie)
    # similarity_matrix, books_rev_ind, book_starting_index = logic(book_description, movie_code_names, movie_reviews)
    data = cosine_sim(movie, book_description, movie_reviews, movie_description)
    print(data)
    return json.dumps(data)


if __name__ == '__main__':
    app.run(debug=True)
