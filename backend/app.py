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
movie_code_names = defaultdict(str)
movie_reviews = defaultdict(str)

with open('books_description.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        book = row['book_title']
        desc = row['description']
        book_description[book] = desc
with open('movie_codes.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        movie_code= row['movie_code']
        movie_name = row['movie_name']
        movie_code_names[movie_code] = movie_name
with open('movie_reviews.csv', 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        movie_code= row['movie_code']
        rev = row['review']
        movie_reviews[movie_code] = rev

movie_list = list(movie_code_names.values())

def logic(book_description, movie_code_names, movie_reviews):
    texts = list(movie_reviews.values()) + list(book_description.values())
    books_rev_ind = {}

    vectorizer = TfidfVectorizer(max_features=500)

    tfidf_vectors = vectorizer.fit_transform(texts)
    book_starting_index = len(movie_reviews.values())
    movie_starting_index = 0
    for i, book in enumerate(book_description.keys()):
        books_rev_ind[i + book_starting_index] = book

    similarity_matrix = cosine_similarity(tfidf_vectors)
    return similarity_matrix, books_rev_ind, book_starting_index

similarity_matrix, books_rev_ind, book_starting_index = logic(book_description, movie_code_names, movie_reviews)

def jaccard(movie, query):
    query = query.lower()
    movie = movie.lower()
    string1 = query.split()
    string2 = movie.split()
    set1 = set(string1)
    set2 = set(string2)

    jaccard_similarity = len(set1.intersection(set2)) / len(set1.union(set2))
    return jaccard_similarity

def find_similar_books(movie_name, similarity_matrix, movie_code_names, book_description, books_rev_ind, book_starting_index, num_books=10):
    
    try:
        movie_index = list(movie_code_names.values()).index(movie_name)
    except ValueError:
        print(f"Movie '{movie_name}' not found.")
        return

    # Get the similarity scores for this movie
    movie_scores = similarity_matrix[movie_index][book_starting_index:]

    # Find the indices of the top-n most similar movies
    book_indices = np.argsort(movie_scores)[-num_books-1:-1][::-1]
    if len(book_indices) == 0:
        print("No similar books found.")
        return

    out = []

    # Print out the names of the top-n most similar movies
    # print(f"Books similar to '{movie_name}':")
    for book_index in book_indices:
        book_index += book_starting_index
        if book_index in books_rev_ind:
            book_name = books_rev_ind[book_index]
            out.append(book_name)
            # out[book_name] = book_description[book_name]
            # print(f"  - {book_name}")
        else:
            continue
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
    movies = request.args.getlist("movies")
    print(movies)
    movie = str(movies[0])
    print(movie)
    # similarity_matrix, books_rev_ind, book_starting_index = logic(book_description, movie_code_names, movie_reviews)
    data = find_similar_books(movie, similarity_matrix, movie_code_names, book_description, books_rev_ind, book_starting_index, num_books=10)
    # return json.dumps([dict(zip(keys, i)) for i in data])
    print(data)
    return json.dumps(data)


if __name__ == '__main__':
    app.run(debug=True)
