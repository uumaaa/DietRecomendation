import json
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def loadData():
    with open('Data/recipes_data.json', 'r') as file:
        recipes_data = json.load(file)
    with open('Data/tagged_diets.json', 'r') as file:
        tagged_diets = json.load(file)
    with open('Data/vector_to_idx.json', 'r') as file:
        vector_to_idx = json.load(file)

    vector_keys = [np.fromstring(key[1:-1], sep=' ') for key in vector_to_idx.keys()]
    vector_values = [vector_to_idx[key] for key in vector_to_idx.keys()]

    embedding_size = len(vector_keys[0])
    vector_keys = np.array([np.pad(vector, (0, embedding_size - len(vector)), 'constant') for vector in vector_keys])

    return recipes_data, tagged_diets, vector_to_idx, vector_keys, vector_values, embedding_size

def createKDTree(vector_keys):
    kdtree = KDTree(vector_keys)
    return kdtree

def get_nearest_recipes(user_vector, kdtree, vector_values, K):
    user_vector = user_vector.reshape(1, -1)

    distances, indices = kdtree.query(user_vector, k=K)

    nearest_recipes = [vector_values[idx] for idx in indices[0]]

    return nearest_recipes, indices[0], distances[0]

def get_user_vector(vector_keys, vector_values, index):
    Index = int(index)

    user_vector = vector_keys[Index] 

    user_vector_description = vector_values[Index]

    return user_vector, user_vector_description

def update_user_vector(user_vector, selected_recipe_vector):

    updated_user_vector = (user_vector + selected_recipe_vector) / 2

    updated_user_vector = updated_user_vector.reshape(1, -1)

    return updated_user_vector

def plot_vectors(vectors, labels, user_vector=None, title="3D Vector Plot"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    
    for vector, label in zip(vectors, labels):
        ax.scatter(vector[0], vector[1], vector[2], label=label)
    
    if user_vector is not None:
        user_vector = user_vector.flatten()
        ax.scatter(user_vector[0], user_vector[1], user_vector[2], color='red', label='User Vector', s=100)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(prop={'size': 6})
    plt.show()

def main():
    # Load data
    recipes_data, tagged_diets, vector_to_idx, vector_keys, vector_values, embedding_size = loadData()
    # Create KDTree
    kdtree = createKDTree(vector_keys)
    # Initial user vector
    user_vector, user_vector_description = get_user_vector(vector_keys, vector_values, 186)
    
    print(f"Initial Recipe: {user_vector_description} - {recipes_data[user_vector_description]}")
    
    while True:
        nearest_recipes, indices, distances = get_nearest_recipes(user_vector, kdtree, vector_values, 20)
        
        for idx, recipe in enumerate(nearest_recipes):
            recipe_index = indices[idx]
            print(f"Recipe {idx}: {recipe} - {recipes_data[recipe]}")
            print(f"Distance: {distances[idx]}\n")
        
        # Plot the vectors
        plot_vectors(vector_keys[indices], nearest_recipes, user_vector, title="3D Vector Plot of Recipes")
        
        # User selects a recipe
        selection = int(input("Select a recipe index or -1 to quit: "))
        if selection == -1:
            break
        
        user_vector = update_user_vector(user_vector, vector_keys[indices[selection]])
        user_vector_description = nearest_recipes[selection]
        print(f"Updated Recipe: {user_vector_description} - {recipes_data[user_vector_description]}")
        
if __name__ == "__main__":
    main()
