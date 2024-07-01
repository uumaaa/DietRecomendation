from flask import Flask, request, jsonify
import json
import numpy as np
from sklearn.neighbors import KDTree

app = Flask(__name__)


# Load the data
def loadData():
    with open("../Data/recipes_data.json", "r") as file:
        recipes_data = json.load(file)
    with open("../Data/tagged_diets.json", "r") as file:
        tagged_diets = json.load(file)
    with open("../Data/vector_to_idx.json", "r") as file:
        vector_to_idx = json.load(file)

    vector_keys = [np.fromstring(key[1:-1], sep=" ") for key in vector_to_idx.keys()]
    vector_values = [vector_to_idx[key] for key in vector_to_idx.keys()]

    embedding_size = len(vector_keys[0])
    vector_keys = np.array(
        [
            np.pad(vector, (0, embedding_size - len(vector)), "constant")
            for vector in vector_keys
        ]
    )

    return (
        recipes_data,
        tagged_diets,
        vector_to_idx,
        vector_keys,
        vector_values,
        embedding_size,
    )


(
    recipes_data,
    tagged_diets,
    vector_to_idx,
    vector_keys,
    vector_values,
    embedding_size,
) = loadData()


# Helper functions
def createKDTree(vector_keys):
    kdtree = KDTree(vector_keys)
    return kdtree


def get_nearest_recipes(user_vector, kdtree, vector_values, K):
    user_vector = user_vector.reshape(1, -1)
    distances, indices = kdtree.query(user_vector, k=K)
    nearest_recipes = [vector_values[idx] for idx in indices[0]]
    return nearest_recipes, indices[0], distances[0]


def update_user_vector(user_vector, selected_recipe_vector):
    updated_user_vector = (user_vector + selected_recipe_vector) / 2
    updated_user_vector = updated_user_vector.reshape(1, -1)
    return updated_user_vector


def parse_vector(vector_string):
    return np.fromstring(vector_string[1:-1], sep=" ")


# Routes
@app.route("/recipes", methods=["GET"])
def get_recipes():
    user_vector_str = request.json["user_vector"]
    user_vector = parse_vector(user_vector_str)
    K = int(request.args.get("K", 20))
    nearest_recipes, indices, distances = get_nearest_recipes(
        user_vector, kdtree, vector_values, K
    )
    return jsonify(
        {
            "nearest_recipes": nearest_recipes,
            "indices": indices.tolist(),
            "distances": distances.tolist(),
        }
    ), 200


@app.route("/recipes", methods=["POST"])
def post_recipe():
    user_vector_str = request.json["user_vector"]
    selected_recipe_index = request.json["selected_recipe_index"]
    user_vector = parse_vector(user_vector_str)
    selected_recipe_vector = vector_keys[selected_recipe_index]

    updated_user_vector = update_user_vector(user_vector, selected_recipe_vector)
    return jsonify({"updated_user_vector": updated_user_vector.tolist()}), 200


kdtree = createKDTree(vector_keys)
if __name__ == "__main__":
    app.run(debug=True)
