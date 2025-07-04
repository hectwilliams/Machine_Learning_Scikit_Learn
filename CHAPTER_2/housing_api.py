import os
import tarfile
import urllib
import urllib.request
import pandas as pd
import numpy as np 
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import pprint
from sklearn.svm import SVR;
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform, truncnorm, t, sem

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz" # https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz 

rooms_idx, bedrooms_idx, population_idx, households_idx = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y = None):
        return self 
    
    def transform(self, X):
        rooms_per_household = X[:, rooms_idx] / X[:, households_idx]
        population_per_household = X[:, population_idx] / X[:, households_idx]
        bedrooms_per_room = X[:, bedrooms_idx] / X[:, rooms_idx]
        new_attr_data = np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]        
        if not self.add_bedrooms_per_room:
            new_attr_data = new_attr_data[:, range(0, new_attr_data.shape[1] - 1) ] # numpy indexing + slice
        return new_attr_data


class SplitDataset(BaseEstimator, TransformerMixin):
   
    def __init__(self, split_ratio = 0.2, random_seed= 42):

        self.split_ratio = split_ratio
        
        self.random_seed = random_seed

    def fit(self, X, y= None):

        return self

    def transform(self, housing_all):

        strat_training_set, strat_test_set = stratified_split_train_test(housing_all, split_ratio=self.split_ratio, random_seed=self.random_seed)

      # drop target (i.e. median house value)
        housing = strat_training_set.drop( columns=["median_house_value"] )

        # copy target (i.e. median house value) 
        self.Y_train  = strat_training_set["median_house_value"].copy()

        self.X_train = strat_training_set.drop(columns=["median_house_value"])

        #copy test set 
        self.X_test = strat_test_set.drop(columns=["median_house_value"])

        self.Y_test = strat_test_set["median_house_value"].copy()

        return self.X_train, self.Y_train, self.X_test, self.Y_test


class ColumnTransform(BaseEstimator, TransformerMixin):
    '''
        tranforms numerical and categorical columns 
    '''

    def __init__(self ):

        pass 

    def fit(self, X, y = None):

        return self 

    def transform(self, dataset):
        
        x_train, y_train, x_test, y_test = dataset
        
        features = list(x_train)

        numerical_attr, categorical_attr = features[:-1] , [features[-1]]

        num_pipeline =  Pipeline(
            [ 

                ('imputer', SimpleImputer(strategy="median")),  # fit + tranform
            
                ("attribs_adder", CombinedAttributesAdder() ),  # tranform
            
                ("std_scaler",  StandardScaler() ) # transform 
            ] 

        )

        cat_pipeline = Pipeline( 
            [
                ('onehot', OneHotEncoder())
            ]
        )

        pipeline = ColumnTransformer([
    
            # transform numercial data
            ("num", num_pipeline, numerical_attr), 
            
            # transform categorical data
            ("cat", cat_pipeline,  categorical_attr),
            
        ])

        # fit x_train required by imputer and std_scaler
        return pipeline.fit_transform(x_train), y_train, pipeline.transform(x_test), y_test, numerical_attr+ categorical_attr+ ['room_per_household', 'pop_per_household', 'bedroom_per_household']



class FilterOutFeatures(BaseEstimator, TransformerMixin):
    '''
        runs random forest and and filters-out unimportant features 
    '''

    def __init__(self):

        pass 

    def fit(self, X, y = None):

        return self 

    def transform(self, dataset):
        
        x_train, y_train, x_test, y_test, attrs = dataset
        
        rand_searc_cv = RandomizedSearchCV (
            
            estimator=RandomForestRegressor(),
            
            scoring= 'neg_mean_squared_error',

            cv = 10,

            param_distributions = {
                'n_estimators': [3, 10, 30],
                    'max_features': [2,3,4],
                    'max_depth': [5,10],
                    'min_samples_split': [2,5,10],
                    'min_samples_leaf': [1,2,4],
                    'bootstrap': [True, False],
            },

            random_state = 24
        )

        rand_searc_cv.fit(x_train, y_train)
        
        model = rand_searc_cv.best_estimator_ # best model 

        feature_important = model.feature_importances_ # this is what we want !
        
        feature_map = sorted(zip( np.round((feature_important * 100),3), attrs), reverse=True)

        remove_features = list( map(lambda x: x[1],filter(lambda ele: ele[0] < 0.8, feature_map)) ) # remove features with importance percentage less than 1 percent

        indices = np.where (np.isin(attrs, remove_features) == True ) 

        if indices:
            x_train = np.delete(x_train, indices, axis = 1)
            x_test = np.delete(x_test, indices, axis = 1)
            attrs = np.delete (attrs, indices, axis = 0)

            pprint.pprint(f'features importance less than 0.8% to be remove: {remove_features}')

        return x_train, y_train.to_numpy(), x_test, y_test.to_numpy(), attrs


cv_grid_search_svr  = GridSearchCV(
            
    estimator = SVR(),

    param_grid = [

    {
        'kernel': ['rbf'],
        
        'C':  [0.2,  0.8],

        'gamma':  ['auto']

    },

    {
    'kernel': ['linear'],

    'C':  [0.2,  0.8],

    'gamma': ['auto']

    }

    ],

    return_train_score = True,

    scoring = 'neg_mean_squared_error',

    cv = 5,

    n_jobs = 5, 

    refit=True

)

cv_random_search_svr  = RandomizedSearchCV(
    
    random_state = 24,
    
    cv = 2, 

    estimator = SVR(),
  
    scoring = 'neg_mean_squared_error',
  
    return_train_score = True,

    param_distributions = {

        'kernel': ['linear', 'rbf'],
            
        'C': np.float32(np.linspace(0.2, 5.0, 10).tolist()),
        
        'epsilon': [0.2],

        'degree': [2,3,4,5,6]
        
        # 'gamma': ['scale', 'auto']
        # 'gamma':   np.linspace(0, 1, 20).tolist(),


    },

    refit = True, 

    n_jobs=5,

    n_iter=5

)

class TrainDataset(BaseEstimator, TransformerMixin):
    """
        train dataset-train
    """

    def fit(self, dataset, y = None):
        
        x_train, y_train, x_test, y_test, attrs = dataset

        cv_random_search_svr.fit(x_train[:10], y_train[:10])

        cv_random_search_svr_dict = cv_random_search_svr.__dict__

        for key in cv_random_search_svr_dict.keys():

            if key[-1] == '_':

                self.__dict__[key] = cv_random_search_svr_dict[key]
        
        return self 

    def transform(self, dataset):
        
        x_train, y_train, x_test, y_test, attrs = dataset

        return x_train, y_train, x_test, y_test, attrs, self.best_estimator_

class PredictDataset(BaseEstimator, TransformerMixin):

    def fit(self, X, y= None):

        return self 
    
    def fit_predict(self, dataset, y=None, conf= None):

        '''
            predict test dataset-test
        '''

        _, _, x_test, y_test, attrs, model= dataset
        
        x_test_predict = model.predict( x_test[:] )

        squared_errors = (y_test - x_test_predict)**2

        conf_interval = t.interval(

            conf,

            len(squared_errors) - 1, 

            loc=squared_errors.mean(), 

            scale= sem(squared_errors)

        )

        final_mse = np.sqrt(mean_squared_error(y_test, x_test_predict))

        print(f'score: {final_mse}')
        
        conf_interval = np.sqrt(conf_interval)
        
        print(f' 95% confidence interval  {conf_interval} ')

        return x_test_predict

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path) # file located at url is archived to local disk   
    housing_tgz = tarfile.open(tgz_path) # open archive file 
    housing_tgz.extractall(  path=housing_path ) 
    housing_tgz.close()  # close archive file 

def load_housing_data(housing_path = HOUSING_PATH):
    sym_link_path_house_csv = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../Data/housing.csv' ) 
    local_path_house_csv =  os.path.join(housing_path, "housing.csv")

    csv_path = local_path_house_csv

    if os.path.islink(sym_link_path_house_csv):
        csv_path = sym_link_path_house_csv
        
    elif not os.path.exists(local_path_house_csv):
        fetch_housing_data()
        
    return pd.read_csv(csv_path) 

def split_train_test(data: pd.DataFrame, test_ratio: float):
    # this would break if dataset was refreshed 
    rng = np.random.default_rng(seed=42)
    shuffled_indices =  rng.permutation(len(data))
    test_set_size = int(len(data) * test_ratio) 
    test_indices = shuffled_indices[ : test_set_size] 
    train_indices = shuffled_indices[ test_set_size : ]
    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(id, ratio):
    # computes hash of each id, returms true if has(id) is lower than 20% of maximum hash value (2**32)
    return crc32( np.int64(id)) & 0xFFFFFFFF < ratio * 2**32 

def split_train_test_by_id(data, ratio, column_attribute):
    # new training data instances must be appended to old datasheet 
    ids = data[column_attribute]
    in_test_set = ids.apply(lambda id: test_set_check(id, ratio))
    return data.loc[~in_test_set], data.loc[in_test_set] # training_set , test_set 

def stratified_split_train_test(data: pd.DataFrame, split_ratio: np.float32, random_seed: np.int32 = 42) -> tuple:
    data["income_cat"] = pd.cut(data["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=range(1, 6))
    split = StratifiedShuffleSplit(n_splits=1, test_size=split_ratio, random_state=random_seed)

    for train_indices, test_indices in split.split(data, data["income_cat"]) :
        strat_train_set = data.loc[train_indices]
        strat_test_set = data.loc[test_indices]
    
    # remove "income_cat" field from both sets
    for data_set_ in (strat_test_set, strat_train_set):
        data_set_.drop('income_cat', axis=1, inplace=True)

    return strat_train_set, strat_train_set


def preprocess():
    
    housing_all = load_housing_data()

    full_pipeline = Pipeline(

        [
            ('split_data', SplitDataset()),
            
            ("column_transform" , ColumnTransform()),

            ("filter_attribs" , FilterOutFeatures()),

            ("fit_training", TrainDataset()),

            ("predict_test", PredictDataset())

        ]

    )

    return full_pipeline.fit_predict(housing_all, predict_test__conf={'conf': 95})

