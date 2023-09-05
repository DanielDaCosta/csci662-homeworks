import argparse
from operator import methodcaller
import string

def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Classifier")
    parser.add_argument("-t", help="path of the input file in the form <text>TAB<label>")
    parser.add_argument("-p", help="path of the prediction file in the form <prediction_label>")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    # True Labels
    with open(args.t) as file:
        data = file.read().splitlines()

    data_split = map(methodcaller("rsplit", "\t", 1), data)
    _, true_labels = map(list, zip(*data_split))

    # Pred Labels
    with open(args.p) as file:
        pred_labels = file.read().splitlines()

    # Check if both files have the same number of samples
    if len(true_labels) != len(pred_labels):
        raise ValueError("Both files must have the same length.")

    # Calculate the accuracy
    total_samples = len(true_labels)
    correct_predictions = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred)
    accuracy_percentage = (correct_predictions/total_samples) * 100

    print(f"Accuracy: {accuracy_percentage:.2f}%")