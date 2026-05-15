from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import json
import os

"""## Load Data"""

ratings_df = pd.read_csv("ml-1m/ratings.dat",
                      delimiter="::",
                      names=["userId", "movieId", "rating", "tstamp"],
                      engine='python')

# data = Dataset.load_builtin('ml-1m')

# ratings_df = pd.DataFrame(data.raw_ratings, columns=["userId", "movieId", "rating", "tstamp"])
rating_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating')
rating_matrix

"""## Create groups"""
# function to create the groups in one of three ways: similar, disimilar, random
def build_groups(group_type='random', num_folds=5, num_groups=100, group_size=8, seed=2025): #note: default is to create random groups
  np.random.seed(seed)

  for fold in range(1, num_folds + 1):
    # Load the training matrix for the current fold
    train_matrix_file = os.path.join("matrices", f"train_matrix_fold_{fold}.csv")
    if not os.path.exists(train_matrix_file):
        print(f"Training matrix for fold {fold} not found. Skipping.")
        continue

    train_matrix = pd.read_csv(train_matrix_file, index_col=0)
    train_matrix = train_matrix.fillna(0)  # Fill NaN values with 0 for similarity calculation

    sim_matrix = cosine_similarity(train_matrix)

    user_ids = train_matrix.index.to_list()
    user_id_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
    groups = []

    for _ in range(num_groups):
      if group_type == 'random':
        group = np.random.choice(user_ids, group_size, replace=False).tolist()

      elif group_type in ['similar', 'diverse']:
        random_user = int(np.random.choice(user_ids)) #Greedy approach, randomly select user and build group around them
        random_idx = user_id_to_index[random_user]
        sim_scores = pd.Series(sim_matrix[random_idx], index=user_ids) #get vector of how similar every user is to random_user

        if group_type == 'similar':
            sorted_users = sim_scores.sort_values(ascending=False)
        elif group_type == 'diverse':
            sorted_users = sim_scores.sort_values(ascending=True)

        top_users = sorted_users[sorted_users.index != random_user].head(group_size - 1).index.tolist() #get list of top users according to what we want group size to be
        group = [random_user] + top_users

      else:
          raise ValueError("group_type must be 'random', 'similar', or 'diverse'")

      groups.append(group)

    groups_file = os.path.join("groups", f"{group_type}_groups-{num_groups}_size-{group_size}_fold_{fold}.json")
    with open(groups_file, 'w') as file:
        json.dump(groups, file)

    print(f"Groups for fold {fold} saved to {groups_file}")


  # # fill NaN vals with 0 for similarity calculation
  # filled_matrix = rating_matrix.fillna(0).values
  # sim_matrix = cosine_similarity(filled_matrix)
  # user_id_to_index = {uid: idx for idx, uid in enumerate(user_ids)}

  # groups = []

  # for _ in range(num_groups):
  #   if group_type == 'random':
  #     group = np.random.choice(user_ids, group_size, replace=False).tolist()

  #   elif group_type in ['similar', 'diverse']:
  #     random_user = int(np.random.choice(user_ids)) #Greedy approach, randomly select user and build group around them
  #     random_idx = user_id_to_index[random_user]
  #     sim_scores = pd.Series(sim_matrix[random_idx], index=user_ids) #get vector of how similar every user is to random_user

  #     if group_type == 'similar':
  #         sorted_users = sim_scores.sort_values(ascending=False)
  #     elif group_type == 'diverse':
  #         sorted_users = sim_scores.sort_values(ascending=True)

  #     top_users = sorted_users[sorted_users.index != random_user].head(group_size - 1).index.tolist() #get list of top users according to what we want group size to be
  #     group = [random_user] + top_users

  #   else:
  #       raise ValueError("group_type must be 'random', 'similar', or 'diverse'")

  #   groups.append(group)


  # return groups

# runtime on T4 High-RAM: 4s
group_size = 8
num_groups = 100
# random_groups = build_groups(relevance_matrix, group_size=group_size, num_groups=num_groups, group_type='random')
# similar_groups = build_groups(relevance_matrix, group_size=group_size, num_groups=num_groups, group_type='similar')
# diverse_groups = build_groups(relevance_matrix, group_size=group_size, num_groups=num_groups, group_type='diverse')

# all_groups = {'random': random_groups, 'similar': similar_groups, 'diverse': diverse_groups}

# for group_type, groups in all_groups.items():
#   filename = f"groups/{group_type}_groups-{num_groups}_size-{group_size}.json"
#   with open(filename, 'w') as file:
#     json.dump(groups, file)
build_groups(group_type='random', num_folds=5, num_groups=num_groups, group_size=group_size, seed=2025)
build_groups(group_type='similar', num_folds=5, num_groups=num_groups, group_size=group_size, seed=2025)
build_groups(group_type='diverse', num_folds=5, num_groups=num_groups, group_size=group_size, seed=2025)
