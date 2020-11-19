import pandas as pd
from sklearn.model_selection import train_test_split

from config import TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE


def load_data():
    """
    @returns x_train, x_validation, x_test, y_train, y_validation, y_test,
    """
    df = pd.read_pickle('train.pickle')
    x_train, x_alt, y_train, y_alt = train_test_split(
        df['comment_text'], df.drop(columns='comment_text'),
        train_size=TRAIN_SIZE,
        test_size=VALIDATION_SIZE + TEST_SIZE,
        random_state=42)
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_alt, y_alt,
        train_size=VALIDATION_SIZE,
        test_size=TEST_SIZE,
        random_state=42)

    return x_train, x_validation, x_test, y_train, y_validation, y_test