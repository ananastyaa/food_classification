import csv

from joblib import load
from sklearn.metrics import accuracy_score
from train import load_data


def main():
    test_csv_path = "prepared/test.csv"
    test_data, labels = load_data(test_csv_path)
    model = load("model/model.joblib")
    predictions = model.predict(test_data)
    accuracy = accuracy_score(labels, predictions)
    metrics = [{"accuracy": accuracy}]
    with open("metrics/accuracy.csv", 'w', newline='') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(metrics)


if __name__ == "__main__":
    main()
