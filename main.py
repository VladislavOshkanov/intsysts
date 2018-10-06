import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
# %matplotlib inline


def preprocessing(df_train, df_test):
    '''
    Parameters
    ----------
    df_train : pandas Dataframe
    df_test : pandas Dataframe
    
    Return
    ----------
    df_train : pandas Dataframe
    df_test : pandas Dataframe
    y_train : pandas Series
    '''
    # remove outliers in GrLivArea
    df_train.drop(df_train[df_train['GrLivArea'] > 4500].index, inplace=True)   

    # Normalize SalePrice using log_transform
    y_train = np.log1p(df_train['SalePrice'])
    # Remove SalePrice from training and merge training and test data
    df_train.pop('SalePrice')
    dataset = pd.concat([df_train, df_test])

    # Numerical variable with "categorical meaning"
    # Cast it to str so that we get dummies later on
    dataset['MSSubClass'] = dataset['MSSubClass'].astype(str)

    
    ### filling NaNs ###
    # no alley
    dataset["Alley"].fillna("None", inplace=True)

    # no basement
    dataset["BsmtCond"].fillna("None", inplace=True)
    dataset["BsmtExposure"].fillna("None", inplace=True)
    dataset["BsmtFinSF1"].fillna(0, inplace=True)               
    dataset["BsmtFinSF2"].fillna(0, inplace=True)               
    dataset["BsmtUnfSF"].fillna(0, inplace=True)                
    dataset["TotalBsmtSF"].fillna(0, inplace=True)
    dataset["BsmtFinType1"].fillna("None", inplace=True)
    dataset["BsmtFinType2"].fillna("None", inplace=True)
    dataset["BsmtFullBath"].fillna(0, inplace=True)
    dataset["BsmtHalfBath"].fillna(0, inplace=True)
    dataset["BsmtQual"].fillna("None", inplace=True)

    # most common electrical system
    dataset["Electrical"].fillna("SBrkr", inplace=True)

    # one missing in test; set to other
    dataset["Exterior1st"].fillna("Other", inplace=True)
    dataset["Exterior2nd"].fillna("Other", inplace=True)

    # no fence
    dataset["Fence"].fillna("None", inplace=True)

    # no fireplace
    dataset["FireplaceQu"].fillna("None", inplace=True)

    # fill with typical functionality
    dataset["Functional"].fillna("Typ", inplace=True)

    # no garage
    dataset["GarageArea"].fillna(0, inplace=True)
    dataset["GarageCars"].fillna(0, inplace=True)
    dataset["GarageCond"].fillna("None", inplace=True)
    dataset["GarageFinish"].fillna("None", inplace=True)
    dataset["GarageQual"].fillna("None", inplace=True)
    dataset["GarageType"].fillna("None", inplace=True)
    dataset["GarageYrBlt"].fillna("None", inplace=True)

    # "typical" kitchen
    dataset["KitchenQual"].fillna("TA", inplace=True)

    # lot frontage (no explanation for NA values, perhaps no frontage)
    dataset["LotFrontage"].fillna(0, inplace=True)

    # Masonry veneer (no explanation for NA values, perhaps no masonry veneer)
    dataset["MasVnrArea"].fillna(0, inplace=True)
    dataset["MasVnrType"].fillna("None", inplace=True)

    # most common value
    dataset["MSZoning"].fillna("RL", inplace=True)

    # no misc features
    dataset["MiscFeature"].fillna("None", inplace=True)

    # description says NA = no pool, but there are entries with PoolArea >0 and PoolQC = NA. Fill the ones with values with average condition
    dataset.loc[(dataset['PoolQC'].isnull()) & (dataset['PoolArea']==0), 'PoolQC' ] = 'None'
    dataset.loc[(dataset['PoolQC'].isnull()) & (dataset['PoolArea']>0), 'PoolQC' ] = 'TA'

    # classify missing SaleType as other
    dataset["SaleType"].fillna("Other", inplace=True)

    # most common
    dataset["Utilities"].fillna("AllPub", inplace=True)

    
    ### feature engineering ###
    # create new binary variables: assign 1 to mode
    dataset["IsRegularLotShape"] = (dataset["LotShape"] == "Reg") * 1
    dataset["IsLandLevel"] = (dataset["LandContour"] == "Lvl") * 1
    dataset["IsLandSlopeGentle"] = (dataset["LandSlope"] == "Gtl") * 1
    dataset["IsElectricalSBrkr"] = (dataset["Electrical"] == "SBrkr") * 1
    dataset["IsGarageDetached"] = (dataset["GarageType"] == "Detchd") * 1
    dataset["IsPavedDrive"] = (dataset["PavedDrive"] == "Y") * 1
    dataset["HasShed"] = (dataset["MiscFeature"] == "Shed") * 1
    # was the house remodeled? if yes, assign 1
    dataset["Remodeled"] = (dataset["YearRemodAdd"] != dataset["YearBuilt"]) * 1
    # assign 1 to houses which were sold the same year they were remodeled
    dataset["RecentRemodel"] = (dataset["YearRemodAdd"] == dataset["YrSold"]) * 1
    # assign 1 to houses which were sold the same year they were built
    dataset["VeryNewHouse"] = (dataset["YearBuilt"] == dataset["YrSold"]) * 1

    
    ### normalization ###
    # normalize distribution for continuous variables with skew > 3
    continuous_vars = ['1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'EnclosedPorch',\
                'GarageArea', 'GrLivArea', 'LotArea', 'LotFrontage', 'MasVnrArea', 'MiscVal',\
                'OpenPorchSF', 'PoolArea', 'ScreenPorch', 'TotalBsmtSF', 'WoodDeckSF']
    skew_threshold = 3
    for entry in continuous_vars:
        if dataset[entry].skew() > skew_threshold:
            dataset[entry] = np.log1p(dataset[entry])
    
    
    ### standardization ###
    # standardization for continuous variables
    sub_df = dataset[continuous_vars]
    array_standard = StandardScaler().fit_transform(sub_df)
    df_standard = pd.DataFrame(array_standard, dataset.index, continuous_vars)
    dataset.drop(dataset[continuous_vars], axis=1, inplace=True)
    dataset = pd.concat([dataset, df_standard], axis=1)
    
    
    ### dummies ###
    # split back to training and test set
    df_train_len = len(df_train)
    df_dummies =  pd.get_dummies(dataset)
    df_train = df_dummies[:df_train_len]
    df_test = df_dummies[df_train_len:]

    return df_train, df_test, y_train

def main():
    df_train = pd.read_csv('/home/voshkanov/house-prices-datasets/train.csv', index_col='Id')
    df_test = pd.read_csv('/home/voshkanov/house-prices-datasets/test.csv', index_col='Id')

    print (df_train.head(5))

    exit
    print('Categorical: ', df_train.select_dtypes(include=['object']).columns)

    #numerical features (see comment about 'MSSubCLass' here above)
    print('Numerical: ', df_train.select_dtypes(exclude=['object']).columns)

    df_train, df_test, y_train_before_split = preprocessing(df_train, df_test)

    # 80/20 split for df_train
    validation_size = 0.2
    seed = 3
    X_train, X_validation, y_train, y_validation = train_test_split(df_train, y_train_before_split, \
                test_size=validation_size, random_state=seed)
    X_test = df_test
    # list of tuples: the first element is a string, the second is an object
    estimators = [('LassoCV', LassoCV()), ('RidgeCV', RidgeCV()),\
                ('RandomForest', RandomForestRegressor()), ('GradientBoosting', GradientBoostingRegressor())]

    for estimator in estimators:
        scores = cross_val_score(estimator=estimator[1],
                                X=X_train,
                                y=y_train,
                                scoring='r2',
                                cv=3,
                                n_jobs=-1)
    #print('CV accuracy scores: %s' % scores)
    print(estimator[0], 'CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    #LassoCV
    gs = GridSearchCV(
                    estimator=LassoCV(),
                    param_grid={'eps':[10**-7, 10**-5, 10**-3],
                                'n_alphas':[25, 50, 75]},
                    scoring='r2',
                    cv=5,
                    n_jobs=-1)

    gs = gs.fit(X_train, y_train)
    print('LassoCV:')
    print('Training accuracy: %.3f' % gs.best_score_)
    print(gs.best_params_)
    est = gs.best_estimator_
    est.fit(X_train, y_train)
    print('Best alpha: ', est.alpha_)
    print('Validation accuracy: %.3f' % est.score(X_validation, y_validation))
main()