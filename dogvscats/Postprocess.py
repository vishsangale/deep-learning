import pandas as pd

from Preprocess import TESTING_SIZE


def create_output_file(predictions):
    predictions_df = pd.DataFrame({'id': range(1, TESTING_SIZE + 1), 'label': predictions[:, 0]})
    predictions_df.to_csv("predictions.csv", index=False)


