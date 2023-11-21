import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        
        self.indices_list = []
        data_length = len(data)
        
        for bag in range(self.num_bags):
            self.indices_list.append(np.random.randint(data_length, size=(data_length)))
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model    = model_constructor()
            indecies = self.indices_list[bag]
            data_bag, target_bag = data[indecies], target[indecies]
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here
        if self.oob:
            self.data   = data
            self.target = target
        
    def predict(self, data):
        prediction = []
        for bag in range(self.num_bags):
            prediction.append(self.models_list[bag].predict(data))
        return np.array(prediction).mean(axis=0)
    
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        self.list_of_predictions_lists = np.array([[self.models_list[j].predict(np.array([self.data[i]]))[0]
            for j in range(len(self.models_list))
                if i not in self.indices_list[j]]
            for i in range(len(self.data))
        ], dtype=object)
        

    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = [np.mean(a) for a in self.list_of_predictions_lists]
        
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        
        return np.nanmean((self.target - self.oob_predictions) ** 2)