from torch.utils.data import Dataset


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
