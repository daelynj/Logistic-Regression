from process_data import ProcessData
from train import Train
from train_kfold import TrainKfold
from train_gaussian import TrainGaussian


def main():
    print('processing data...')
    pd = ProcessData()
    data, half_data = pd()
    print('done processing!')

    trainer = Train()
    kfold_trainer = TrainKfold()
    gaussian_trainer = TrainGaussian()

    # [0.8795, 0.9035, 0.9285, 0.9435, 0.9505, 0.96, 0.963, 0.958, 0.954, 0.9525]
    trainer(data, 'lr', 0.0001, 4, "logistic regression")
    # [0.957, 0.959, 0.96, 0.9595, 0.9575, 0.958, 0.9595, 0.953, 0.9515, 0.947]
    trainer(half_data, 'svm', 0.04, 1.6, "support vector machine")

    # lr: 1.0248700000000002 - training: 0.9738333333333333 testing: 0.9585
    kfold_trainer(data, 'lr', 0.7, 1.1, k=5)
    # svm: 0.06 - training: 0.9683333333333334 testing: 0.958
    kfold_trainer(half_data, 'svm', 0.04, 1.5, k=5)

    # [0.0026414886541018556, 0.0030797496623704845, 0.0035907244833864723, 0.004186477385849299, 0.004881074274375396,
    # 0.0056909147897226675, 0.006635119509224956, 0.007735981389354633, 0.009019492109107729, 0.010515955741336555,
    # 0.012260704240994024, 0.014294931643181576, 0.016666666666666666, 0.019431906686330536, 0.022655939847975447,
    # 0.02641488654101857]
    gaussian_trainer(half_data, 0.04, 1.5)


if __name__ == '__main__':
    main()
