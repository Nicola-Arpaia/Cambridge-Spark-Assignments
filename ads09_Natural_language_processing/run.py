import argparse
import os
import spacy
import pickle
import pandas as pd
from model import build_model

X_TRAIN_NAME = "X_train.csv"
Y_TRAIN_NAME = "y_train.csv"
X_TEST_NAME = "X_test.csv"

DATA_DIR = "data"
PICKLE_NAME = "model.pickle"

def train_model():
    X = pd.read_csv(os.sep.join([DATA_DIR, X_TRAIN_NAME]))
    y = pd.read_csv(os.sep.join([DATA_DIR, Y_TRAIN_NAME]))

    model = build_model()
    model.fit(X, y)

    # Save to pickle
    with open(PICKLE_NAME, "wb") as f:
        pickle.dump(model, f)


def test_model():
    X = pd.read_csv(os.sep.join([DATA_DIR, X_TEST_NAME]))

    # Load pickle
    with open(PICKLE_NAME, "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X)
    print("### Your predictions ###")
    print(preds)


def main():
    parser = argparse.ArgumentParser(
        description="A command line-tool to manage the project."
    )
    parser.add_argument(
        "stage",
        metavar="stage",
        type=str,
        choices=["train", "test"],
        help="Stage to run.",
    )

    stage = parser.parse_args().stage

    if stage == "train":
        print("Training model...")
        train_model()

    elif stage == "test":
        print("Testing model...")
        test_model()


if __name__ == "__main__":
    main()
