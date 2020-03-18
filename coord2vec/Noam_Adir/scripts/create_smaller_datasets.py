import pandas as pd

server_csv_dir = "/data/home/morpheus/coord2vec_noam/coord2vec/evaluation/tasks/house_pricing"
full_dataset_csv_path = f"{server_csv_dir}/Housing price in Beijing.csv"
small_dataset_csv_path = f"{server_csv_dir}/Housing price in Beijing small.csv"
medium_dataset_csv_path = f"{server_csv_dir}/Housing price in Beijing medium.csv"

n_examples_in_small = 1000
n_examples_in_medium = 10000

full_dataset_df = pd.read_csv(full_dataset_csv_path, engine='python')
small_dataset_df = full_dataset_df[:n_examples_in_small]
medium_dataset_df = full_dataset_df[:n_examples_in_medium]

small_dataset_df.to_csv(small_dataset_csv_path)
medium_dataset_df.to_csv(medium_dataset_csv_path)
