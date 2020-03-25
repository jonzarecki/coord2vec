from torch.utils.data import Dataset
from preprocess import *
from base_pipeline import *

class Feature_Dataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.y[idx]
        sample = (x_sample, y_sample)
        return sample


def save_to_pickle_features():
    cleaned_csv_features = extract_and_filter_csv_data([clean_floor_col, clean_constructionTime_col])
    all_features = extract_geographical_features(cleaned_csv_features)
    pickle_out_features = open("features.pickle", "wb")
    pickle.dump(all_features, pickle_out_features)
    pickle_out_features.close()


# save_to_pickle_features()


def load_from_pickle_features():
    pickle_in_features = open("features.pickle", "rb")
    features = pickle.load(pickle_in_features)
    # print(features)
    pickle_in_features.close()
    return features
