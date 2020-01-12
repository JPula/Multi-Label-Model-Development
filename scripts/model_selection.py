from sklearn.model_selection import train_test_split

def default_split(X, y):
    return train_test_split(X, y, test_size = 0.33)