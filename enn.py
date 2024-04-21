import pandas as pd
from sklearn.neighbors import NearestNeighbors

def enn(input_file, output_file):
    # Read the CSV file
    data = pd.read_csv(input_file)

    # Separate features (X) and labels (y)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Use Nearest Neighbors to identify noise points
    neigh = NearestNeighbors(n_neighbors=3)
    neigh.fit(X)
    distances, indices = neigh.kneighbors(X)
    noise_indices = []
    for i in range(len(indices)):
        if len(set(y[indices[i]]) - {y.iloc[i]}) == 0:
            noise_indices.append(i)

    # Drop noise points from the dataset
    edited_data = data.drop(noise_indices)

    # Write the edited data to a new CSV file
    edited_data.to_csv(output_file, index=False)

# Example usage
    enn("normalized_iris.csv", "enn_iris.csv")
enn("normalized_letter_recognition.csv", "enn_edited_letter_recognition.csv")
