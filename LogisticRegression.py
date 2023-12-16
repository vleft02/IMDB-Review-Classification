class LogisticRegression():
   

    def __init__(self,epochs=20,learning_rate=0.01):
        self.epochs=epochs;
        self.learning_rate=learning_rate;
        self.weights = np.array([])
        
    

    def sigmoid(t):
        return 1/(1+np.exp(-t))


    def fit(self,x_train_input,y_train_input):
        ''' '''
        feature_vector_length = x_train_input.shape[1]
        split_index = int(0.8 * x_train_input.shape[0])  # 80% for training, 20% for dev

        x_train, y_train = x_train_input[:split_index], y_train_input[:split_index]
        x_dev, y_dev= x_train_input[split_index:], y_train_input[split_index:]
        
        # x_train = np.insert(x_train, 0, 1, axis=1)
        print(x_train.shape)
        print(x_dev.shape)
        self.weights = self.initializeWeights(feature_vector_length)
        
        for epoch in range(self.epochs):
            for i in range(x_train.shape(0)):
                print("");
                # for weightIndex in range(self.weights):
                    
                #     currentWeight = self.weights[weightIndex]
                    
                #     weightChange = self.learning_rate * 
                # self.weights[weightIndex] = currentWeight + weightChange


        
    
    def predict(self, x_test):
        recall, precision = 0
        return (recall, precision)

    def SGC(self,w):
        # return w + self.learning_rate*1
        print("parameter estimation")

    @staticmethod
    def initializeWeights(size):
        return np.random.randn(size) * 0.01
    
    def logLikelyhood():
        sum = 0 