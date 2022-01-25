## author: CHU Shaocheng


from sklearn.datasets import load_iris
import scipy.stats as stats
import numpy as np
class NBC:
    def __init__(self,feature_types,num_classes):
        self.features_type = feature_types
        self.num_classes = num_classes
        self.LABELS = [0, 1, 2]
        self.prior = None
        self.mean = None
        self.std = None
        self.posterior = None

    def load_data(self):
        iris = load_iris()
        X, y = iris['data'], iris['target']
        return X, y

    def split_data(self):
        X, y = self.load_data()
        N, D = X.shape
        indices = np.random.permutation(N)
        Xtrain = X[indices[:int(0.8 * N)]]
        ytrain = y[indices[:int(0.8 * N)]]
        Xtest = X[indices[int(0.8 * N):]]
        ytest = y[indices[int(0.8 * N):]]
        return Xtrain, ytrain, Xtest, ytest

    def fit(self, X, y):
        N, D = X.shape

        # Compute π_c
        print("Compute πc:")
        self.calculate_prior(X, y)

        # Compute conditional distributions
        print("\nCompute all conditional distributions of four features:")
        self.calculate_mu_sigma(X,y)

    def predict(self, X):
        self.calculate_posterior(X)
        return np.argmax(self.posterior, axis=0)

    def calculate_mu_sigma(self, X, y):
        N, D = X.shape
        mean = np.zeros((self.num_classes, D))
        std = np.zeros((self.num_classes, D))
        for i in self.LABELS:
            mean[i] = np.mean(X[y == i], axis=0)
            std[i] = np.std(X[y == i], axis=0)
        # non-zero std
        std[std == 0] = 0.000001
        self.mean = mean
        self.std = std

    def calculate_prior(self, X, y):
        N, D = X.shape
        prior = np.zeros(self.num_classes)
        for i in self.LABELS:
            prior[i] = (y == i).sum() / N
        self.prior = prior

    # log sum to prevent underflow
    def log_space_multplication(self,a,b):
        a[a == 0] = 0.000001
        b[b == 0] = 0.000001
        return np.exp(np.log(a) + np.log(b))

    def calculate_posterior(self, X):
        N, D = X.shape
        posterior = np.zeros((self.num_classes,N))
        for i in range(self.num_classes):
            likelihood = np.ones(N)
            for j in range(D):
                likelihood = self.log_space_multplication(likelihood, stats.norm.pdf(X[:, j], self.mean[i, j], self.std[i, j]))
            posterior[i, :] = likelihood * self.prior[i]
        self.posterior = posterior


if __name__ == '__main__':
    nbc = NBC(feature_types=['r', 'r', 'r', 'r'], num_classes=3)

    Xtrain, ytrain, Xtest, ytest = nbc.split_data()

    nbc.fit(Xtrain, ytrain)

    ytrain_hat = nbc.predict(Xtrain)
    ytest_hat = nbc.predict(Xtest)
    print(f"\nPrediction of Xtest points:\n{ytest_hat}")

    train_accuracy = np.mean(ytrain_hat == ytrain)
    print("\nTrain Accuracy: " + str(train_accuracy))
    test_accuracy = np.mean(ytest_hat == ytest)
    print("\nTest Accuracy: "+str(test_accuracy))
