import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from matplotlib.colors import ListedColormap
import os

plt.style.use("fivethirtyeight")


def prepare_data(df):
    X = df.drop("y", axis=1)
    y = df["y"]
    return X, y


def save_model(model, filename):
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)  # ONLY create if model doesn't exist
    filepath = os.path.join(model_dir, filename)  # model / filename
    joblib.dump(model, filepath)
