from train_kfold import TrainKfold
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np


class TrainGaussian:
    def __call__(self, data, c_0, a):
        train, test = data
        x_train, y_train = train
        x_test, y_test = test
        kfold_trainer = TrainKfold()

        cs = []
        gs = []
        for i in np.logspace(1.2, 2.2, num=16):
            print(f'pass: {i}')
            gamma = 1 / (len(x_train) / i)
            gs.append(gamma)
            results = []
            for j in range(5, 15):
                C = c_0 * a ** (2 * j)
                result = kfold_trainer.kfold(
                    train, 'svm gaussian', C, k=5, gamma=gamma)
                results.append(result)
                print(f'result: {result}')

            best = results.index(max(results))
            cs.append(c_0 * a ** (2 * best))

        train_scores, test_scores = [], []
        for g, c in zip(gs, cs):
            clf = SVC(kernel='rbf', C=c, gamma=g).fit(x_train, y_train)
            train_scores.append(clf.score(x_train, y_train))
            test_scores.append(clf.score(x_test, y_test))

        # cs = [0.30375, 0.6834375, 0.455625, 0.455625, 0.455625, 0.6834375]
        print(cs)
        print(gs)

        fig, ax = plt.subplots()
        ax.set_xlabel("gamma")
        ax.set_ylabel("accuracy")
        ax.set_title('Gamma vs Accuracy for Test and Training Sets')
        ax.plot(gs, train_scores, marker='o', label="train")
        ax.plot(gs, test_scores, marker='o', label="test")
        ax.legend()
        plt.savefig('images/gamma.png')
