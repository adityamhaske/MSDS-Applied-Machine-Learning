import sys

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


# download dataset first
def load_dataset():
    df = pd.read_pickle('train100c5k_v2.pkl')
    data = df['data'].values
    target = df['target'].values
    print(data.shape)
    print(target.shape)
    flatten_data = [x.reshape(28 * 28, ) for x in data]
    return flatten_data, target


def demo_knn():
    data, target = load_dataset()
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(data, target)
    accuracy = model.score(data, target)
    print('training accuracy on training set:', accuracy)
    accuracy = model.score(data[:5000], target[:5000])
    print('training accuracy on the 1st category:', accuracy)


def baseline_knn():
    data, target = load_dataset()
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(data, target)

    df = pd.read_pickle('test100c5k_nolabel.pkl')
    test_image = [x.reshape(28 * 28, ) for x in df['data'].values]
    result = model.predict(test_image)
    with open('project_xiaoq.txt', 'w') as file:  # edit here as your username
        file.write('\n'.join(map(str, result)))
        file.flush()


def run(test_image):
    data, target = load_dataset()
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(data, target)

    data = [x.reshape(28 * 28, ) for x in test_image]
    result = model.predict(data)
    with open('project_xiaoq.txt', 'w') as file:  # edit here as your username
        file.write('\n'.join(map(str, result)))
        file.flush()
        return True
    return False


if __name__ == "__main__":
    df = pd.read_pickle(sys.argv[1])  # test set path
    try:
        error = run(df['data'].values)
    except RuntimeError:
        print("An RuntimeError occurred")
    except:
        print("An exception occurred")
