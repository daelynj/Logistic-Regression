from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.svm import SVC


class TrainKfold:
    def __call__(self, data, classifier, c_0, a, k=5, iterations=10):
        train, test = data

        results = []
        print('training...')
        for i in range(iterations):
            print(f'iteration {i}')
            results.append(self.kfold(train, classifier, c_0 * a ** i, k))

        best = results.index(max(results))

        self.__verfify_kfold(data, classifier, c_0 * a ** best)

    def __verfify_kfold(self, data, classifier, C):
        train, test = data
        x_train, y_train = train
        x_test, y_test = test

        clf = None
        if classifier == 'lr':
            clf = LogisticRegression(penalty='l2', max_iter=200,
                                     C=C).fit(x_train, y_train)
        else:
            clf = SVC(kernel='linear', C=C).fit(x_train, y_train)

        train_score = clf.score(x_train, y_train)
        test_score = clf.score(x_test, y_test)

        print(f'{classifier}: {C} - training: {train_score} testing: {test_score}')
        return (train_score, test_score)

    def kfold(self, data, classifier, C, k, gamma=None):
        x_train, y_train = data

        x_chunks = np.split(np.array(x_train), k)
        y_chunks = np.split(np.array(y_train), k)

        scores = []
        for i in range(k):
            print(i)
            xi_train = []
            yi_train = []

            for j in range(k):
                if j == i:
                    continue
                xi_train.append(x_chunks[j])
                yi_train.append(y_chunks[j])

            xi_train = list(self.__flat(xi_train))
            yi_train = list(self.__flat(yi_train))

            xi_test = list(x_chunks[i])
            yi_test = list(y_chunks[i])

            clf = None
            if classifier == 'lr':
                clf = LogisticRegression(
                    penalty='l2', max_iter=200, C=C).fit(xi_train, yi_train)
            elif classifier == 'svm':
                clf = SVC(kernel='linear', C=C).fit(xi_train, yi_train)
            else:
                clf = SVC(kernel='rbf', C=C, gamma=gamma).fit(
                    xi_train, yi_train)

            scores.append(clf.score(xi_test, yi_test))

        return sum(scores) / len(scores)

    def __flat(self, l):
        return [i for s in l for i in s]
