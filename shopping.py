import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    print(len(evidence))
    print(len(labels))
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    get_month_num = {"Jan":0, "Feb":1, "Mar":2, "Apr":3, "May":4, "June":5, "Jul":6, "Aug":7,
                    "Sep":8, "Oct":9, "Nov":10, "Dec":11}
    int_index = {0, 2, 4, 11, 12, 13, 14}

    evidence = []
    labels = []
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            new_evidence = []
            for index, value in enumerate(row):
                if index in int_index:
                    new_evidence.append(int(value))
                elif index == 10:
                    num = get_month_num[value]
                    new_evidence.append(num)
                elif index == 15:
                    if value == "Returning_Visitor":
                        new_evidence.append(1)
                    else:
                        new_evidence.append(0)
                elif index == 16:
                    if value == "FALSE":
                        new_evidence.append(0)
                    else:
                        new_evidence.append(1)
                elif index == 17:
                    if value == "FALSE":
                        labels.append(0)
                    else:
                        labels.append(1)
                else:
                    new_evidence.append(float(value))
            evidence.append(new_evidence)
    return (evidence, labels)

    


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(evidence, labels)
    return knn

    



def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    total_true = 0
    total_false = 0
    correctly_true = 0
    correctly_false = 0
    for label, prediction in zip(labels, predictions):
        if label == 1:
            total_true += 1
            if prediction == label:
                correctly_true += 1
        else:
            total_false += 1
            if prediction == label:
                correctly_false += 1
    sensitivity = correctly_true / total_true
    specificity = correctly_false / total_false
    return (sensitivity, specificity)
        






if __name__ == "__main__":
    main()
