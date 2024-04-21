import csv
import pandas as pd
import numpy as np

class Attribute:
    def __init__(self, name):
        self.name = name
        self.data = []

    def add_data(self, value):
        self.data.append(value)

class Element:
    def __init__(self, index):
        self.index = index
        self.attributes = []

    def add_attribute(self, attribute):
        self.attributes.append(attribute)

def nearest_neighbour(element, cs):
    distances = [sum((element.attributes[m].data[0] - cs[i].attributes[m].data[0])**2 for m in range(len(element.attributes) - 1)) for i in range(len(cs))]
    index = np.argmin(distances)
    return index

def ib2(input_file, output_file):
    all_attributes = []
    all_elements = []

    # Load data from CSV
    data = pd.read_csv(input_file)

    # Extract attributes
    for column in data.columns[:-1]:
        all_attributes.append(Attribute(column))

    # Create elements
    for index, row in data.iterrows():
        element = Element(index)
        for i, value in enumerate(row[:-1]):
            element.add_attribute(Attribute(all_attributes[i].name))
            element.attributes[i].add_data(value)
        element.add_attribute(Attribute(data.columns[-1]))
        element.attributes[-1].add_data(row[data.columns[-1]])
        all_elements.append(element)

    CS = []
    TS = list(all_elements)

    # Moving the first element of TS to CS
    CS.append(all_elements[0])
    TS.pop(0)

    # For all elements of TS find the nearest at CS
    while TS:
        index_IB2 = nearest_neighbour(TS[0], CS)

        # Compare the classes
        if TS[0].attributes[-1].data[0] != CS[index_IB2].attributes[-1].data[0]:
            CS.append(TS[0])

        TS.pop(0)

    # Display results
    print("After applying the IB2 algorithm at the normalized dataset, these are the results:\n")
    print("------------------------CS (" + str(len(CS)) + " elements)------------------------")
    for element in CS:
        for attribute in element.attributes[:-1]:
            print(attribute.data[0], end=", ")
        print(element.attributes[-1].data[0])

    print("\n------------------------TS (" + str(len(TS)) + " elements)------------------------")
    for element in TS:
        for attribute in element.attributes[:-1]:
            print(attribute.data[0], end=", ")
        print(element.attributes[-1].data[0])

    if not TS:
        print("TS is empty :)")

    # Writing the reduced elements to a new CSV file
    try:
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data.columns)  # Write header

            for element in CS:
                writer.writerow([attribute.data[0] for attribute in element.attributes])

        print("\nYou can find the reduced dataset at:", output_file, "that was just created\n\n\n")

    except Exception as e:
        print(e)

# Call the function with the specified file names
ib2("Normalize\normalized_iris.csv", "IB2\ib2reduced_iris.csv")
ib2("Normalize\normalized_letter_recognition.csv", "IB2\ib2reduced_iris.csv")
