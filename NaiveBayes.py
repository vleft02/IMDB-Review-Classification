import numpy as np

class NaiveBayes: 
    def __init__(self):
            self.probabilityC1= None #probabilityC0 = 1 -probabilityC1
            self.boundProbabilityC0 = None 
            self.boundProbabilityC1 = None

    def fit(self, x_train_input, y_train_input): 
        split_index = int(0.8 * x_train_input.shape[0])  # 80% for training, 20% for dev
        x_train, y_train = x_train_input[:split_index], y_train_input[:split_index]
        x_dev, y_dev= x_train_input[split_index:], y_train_input[split_index:]

        numberOfExamples = x_train.shape[0]
        numberOfFeatures = x_train.shape[1]
        x_train = x_train.toarray()
        #y_train = y_train.flatten()

        positiveReviews = 0
        for label in y_train: 
            if label == 1: 
                positiveReviews += 1 

        self.probabilityC1 = positiveReviews / len(y_train)

        '''
        C0: P( X(i) = 0 | C = 0) and C1 = P ( X(i) = 0 | C = 1)
        P( X(i) = 1 | C = 0) = 1 - P( X(i) = 0 | C = 0) and P( X(i) = 1 | C = 1) = P( X(i) = 0 | C = 1)
        '''
        self.boundProbabilityC0 = np.zeros(numberOfFeatures)  
        self.boundProbabilityC1 = np.zeros(numberOfFeatures)
        for i in range(numberOfFeatures): 

            #For every word that is not on a review add to the array correspnding to the review result of the example 
            for j in range(numberOfExamples):  
                if y_train[j] == 0 and x_train[j][i] == 0: 
                    self.boundProbabilityC0[i] += 1 
                elif y_train[j] == 1 and x_train[j][i] == 0: 
                    self.boundProbabilityC1[i] += 1 
        """
        Adding Laplace estimator with an alpha value of 1 
        """
        self.boundProbabilityC0 = [x + 1 / (numberOfExamples - positiveReviews + x) for x in self.boundProbabilityC0] 
        self.boundProbabilityC1 = [x + 1 / (positiveReviews+ x) for x in self.boundProbabilityC1]

    def predict(self, x_test, y_test):
        prediction = list()
        c1 = self.boundProbabilityC1
        c0 = self.boundProbabilityC0
        numberOfFeatures = x_test.shape[1]
        x_test = x_test.toarray()

        for x in range(x_test.shape[0]): 
            positiveReviewProbability = self.probabilityC1 
            negativeReviewProbability = (1 - self.probabilityC1)

            for y in range(numberOfFeatures):
                if x_test[x][y] == 0 and y_test[x] == 0: 
                    negativeReviewProbability = round(negativeReviewProbability * self.boundProbabilityC0[y], 2)
                elif  x_test[x][y] == 1 and y_test[x] == 0: 
                    negativeReviewProbability = round(negativeReviewProbability * (1 - self.boundProbabilityC0[y]), 2)
                elif  x_test[x][y] == 0 and y_test[x] == 1:
                    positiveReviewProbability = round(positiveReviewProbability * self.boundProbabilityC1[y], 2)
                elif  x_test[x][y] == 1 and y_test[x] == 1:
                    positiveReviewProbability = round(positiveReviewProbability * (1 - self.boundProbabilityC1[y]), 2)

            if positiveReviewProbability > negativeReviewProbability: 
                prediction.append(1)
            elif negativeReviewProbability > positiveReviewProbability:
                prediction.append(0)

        return np.array(prediction)
    













    class NaiveBayes: 
    def __init__(self):
            self.probabilityC1= None #probabilityC0 = 1 -probabilityC1
            self.boundProbabilityC0 = None 
            self.boundProbabilityC1 = None

    def fit(self, x_train_input, y_train_input): 
        split_index = int(0.8 * x_train_input.shape[0])  # 80% for training, 20% for dev
        x_train, y_train = x_train_input[:split_index], y_train_input[:split_index]
        x_dev, y_dev= x_train_input[split_index:], y_train_input[split_index:]

        numberOfExamples = x_train.shape[0]
        numberOfFeatures = x_train.shape[1]
        x_train = x_train.toarray()
        #y_train = y_train.flatten()

        positiveReviews = 0
        for label in y_train: 
            if label == 1: 
                positiveReviews += 1 

        self.probabilityC1 = positiveReviews / len(y_train)

        '''
        C0: P( X(i) = 0 | C = 0) and C1 = P ( X(i) = 0 | C = 1)
        P( X(i) = 1 | C = 0) = 1 - P( X(i) = 0 | C = 0) and P( X(i) = 1 | C = 1) = P( X(i) = 0 | C = 1)
        '''
        self.boundProbabilityC0 = np.zeros(numberOfFeatures)  
        self.boundProbabilityC1 = np.zeros(numberOfFeatures)
        for i in range(numberOfFeatures): 

            #For every word that is not on a review add to the array correspnding to the review result of the example 
            for j in range(numberOfExamples):  
                if y_train[j] == 0 and x_train[j][i] == 0: 
                    self.boundProbabilityC0[i] += 1 
                elif y_train[j] == 1 and x_train[j][i] == 0: 
                    self.boundProbabilityC1[i] += 1 
        """
        Adding Laplace estimator with an alpha value of 1 
        """
        self.boundProbabilityC0 = [x + 1 / (numberOfExamples - positiveReviews + x) for x in self.boundProbabilityC0] 
        self.boundProbabilityC1 = [x + 1 / (positiveReviews+ x) for x in self.boundProbabilityC1]
        print("Done!")

    def predict(self, x_test, y_test):
        prediction = list()
        c1 = self.boundProbabilityC1
        c0 = self.boundProbabilityC0
        numberOfFeatures = x_test.shape[1]
        x_test = x_test.toarray()

        for x in range(x_test.shape[0]): 
            positiveReviewProbability = self.probabilityC1 
            negativeReviewProbability = (1 - self.probabilityC1)

            for y in range(numberOfFeatures):
                if x_test[x][y] == 0 and y_test[x] == 0: 
                    negativeReviewProbability = round(negativeReviewProbability * self.boundProbabilityC0[y], 2)
                elif  x_test[x][y] == 1 and y_test[x] == 0: 
                    negativeReviewProbability = round(negativeReviewProbability * (1 - self.boundProbabilityC0[y]), 2)
                elif  x_test[x][y] == 0 and y_test[x] == 1:
                    positiveReviewProbability = round(positiveReviewProbability * self.boundProbabilityC1[y], 2)
                elif  x_test[x][y] == 1 and y_test[x] == 1:
                    positiveReviewProbability = round(positiveReviewProbability * (1 - self.boundProbabilityC1[y]), 2)

            if positiveReviewProbability > negativeReviewProbability: 
                prediction.append(1)
            elif negativeReviewProbability > positiveReviewProbability:
                prediction.append(0)

        return np.array(prediction)
    





for y in range(numberOfFeatures): 
                if x[y] == 1:
                    if self.boundProbabilityC1[y] == 1: 
                        self.boundProbabilityC1[y] = 0.9999999999999999
                    positiveReviewProbability += log(1 - self.boundProbabilityC1[y])
                    negativeReviewProbability += log(1 - self.boundProbabilityC1[y])
                elif x[y] == 0: 
                    positiveReviewProbability += log(self.boundProbabilityC1[y])
                    negativeReviewProbability +=log(self.boundProbabilityC0[y])
