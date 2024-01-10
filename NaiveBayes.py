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

        positiveReviews = 0
        for label in y_train: 
            if label == 1: 
                positiveReviews += 1 

        self.probY = positiveReviews / len(y_train)

        '''
        C0: P( X(i) = 0 | C = 0) and C1 = P ( X(i) = 0 | C = 1)
        P( X(i) = 1 | C = 0) = 1 - P( X(i) = 0 | C = 0) and P( X(i) = 1 | C = 1) = P( X(i) = 0 | C = 1)
        '''
        self.boundProbabilityC0, self.boundProbabilityC1 = np.zeros(numberOfFeatures)  
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
        self.boundProbabilityC1 = [x + 1 / (positiveReviews+ x) for x in self.boundProbabilityc1]

    def predict(self, x_dev, y_dev):
        prediction = list()
        c1 = self.boundProbabilityC1
        c0 = self.boundProbabilityC0
        numberOfFeatures = x_dev.shape[1]

        for x in x_dev: 
            positiveReviewProbability = self.probabilityC1 
            negativeReviewProbability = (1 - self.probabilityC1)

            for y in range(numberOfFeatures):
                if x_dev[x][y] == 0 and y_dev[x] == 0: 
                    negativeReviewProbability = negativeReviewProbability * self.boundProbabilityC0[y]
                elif  x_dev[x][y] == 1 and y_dev[x] == 0: 
                    negativeReviewProbability = negativeReviewProbability * (1 - self.booundProbabilityC0[y])
                elif  x_dev[x][y] == 0 and y_dev[x] == 1:
                    positiveReviewProbability = positiveReviewProbability * self.boundProbabilityC1
                elif  x_dev[x][y] == 1 and y_dev[x] == 1:
                    positiveReviewProbability = positiveReviewProbability * (1 - self.boundProbabilityC1)

            if positiveReviewProbability > negativeReviewProbability: 
                prediction.append(1)
            elif negativeReviewProbability > positiveReviewProbability:
                prediction.append(0)

        return np.array(prediction)