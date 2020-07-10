import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class Train:
    def __call__(self, data, classifier, c_0, a, name, iterations=10):
        train, test = data
        x_train, y_train = train
        x_test, y_test = test

        c, train_scores, test_scores = [], [], []
        for i in range(iterations):
            print(i)
            C = c_0 * a ** i

            clf = None
            if classifier == 'lr':
                clf = LogisticRegression(
                    penalty='l2', C=C).fit(x_train, y_train)
            else:
                clf = SVC(kernel='linear', C=C).fit(x_train, y_train)

            train_scores.append(clf.score(x_train, y_train))
            test_scores.append(clf.score(x_test, y_test))
            c.append(C)

        print(f'train: {train_scores}')
        print(f'test: {test_scores}')

        self.__plot(
            c,
            train_scores,
            test_scores,
            f'{name.capitalize()} Accuracy vs Regularization',
            f'images/{name}.png'
        )

    def __plot(self, x_axis, train_scores, test_scores, title, name):
        fig, ax = plt.subplots()
        ax.set_xlabel("regularization strength")
        ax.set_ylabel("accuracy")
        ax.set_title(title)
        ax.plot(x_axis, train_scores, marker='o', label="train")
        ax.plot(x_axis, test_scores, marker='o', label="test")
        ax.legend()
        plt.savefig(name)
