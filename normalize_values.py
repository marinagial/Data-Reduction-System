import pandas as pd

def normalize_values(input_file, output_file):
    data = pd.read_csv(input_file)
    
    # Exclude the class attribute from normalization
    numeric_columns = data.select_dtypes(include=[float, int]).columns
    data[numeric_columns] = (data[numeric_columns] - data[numeric_columns].min()) / (data[numeric_columns].max() - data[numeric_columns].min())

    # Write the normalized DataFrame to a new CSV file
    data.to_csv(output_file, index=False)

# Example usage:
normalize_values("iris.csv", "normalized_iris.csv")
normalize_values("letter-recognition.csv", "normalized_letter_recognition.csv")

