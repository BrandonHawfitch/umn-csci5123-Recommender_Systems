from sklearn.model_selection import KFold,train_test_split
import numpy as np
import pandas as pd
import json

"""## Load Data"""

ratings_df = pd.read_csv("ml-1m/ratings.dat",
                      delimiter="::",
                      names=["userId", "movieId", "rating", "tstamp"],
                      engine='python')

# Initialize KFold for cross-validation
NUM_FOLDS = 5
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kf.split(ratings_df), start=1):
  # Split ratings_df into train and test sets based on indices
  train_ratings = ratings_df.iloc[train_idx]
  test_ratings = ratings_df.iloc[test_idx]

  # Create train and test matrices
  train_matrix = train_ratings.pivot(index='userId', columns='movieId', values='rating')
  test_matrix = test_ratings.pivot(index='userId', columns='movieId', values='rating')

  # Save train and test matrices for the current fold
  train_filename = f"matrices/train_matrix_fold_{fold}.csv"
  test_filename = f"matrices/test_matrix_fold_{fold}.csv"

  train_matrix.to_csv(train_filename)
  test_matrix.to_csv(test_filename)

  print(f"Saved train matrix to {train_filename} and test matrix to {test_filename}")