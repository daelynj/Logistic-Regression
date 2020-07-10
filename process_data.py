from mnist_reader import load_mnist


class ProcessData:
    def __call__(self):
        x_train, y_train = load_mnist('data', kind='train')
        x_test, y_test = load_mnist('data', kind='t10k')

        x_train, y_train = self.__filter_data(x_train, y_train)
        x_test, y_test = self.__filter_data(x_test, y_test)

        mid = int(len(x_train) / 2)
        return ((x_train, y_train), (x_test, y_test)), ((x_train[:mid], y_train[:mid]), (x_test, y_test))

    def __filter_data(self, x, y):
        x.tolist()
        y.tolist()

        temp_x, temp_y = [], []
        for i in range(len(x)):
            if y[i] in [5, 7]:
                temp_x.append(x[i] / 255)
                temp_y.append(0 if y[i] == 5 else 1)
        return temp_x, temp_y
