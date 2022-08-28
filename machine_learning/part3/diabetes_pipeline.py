##################################################
# End-to-End Diabetes Machine Learning Pipeline II
##################################################


##################################################
# Import İşlemleri
##################################################

import warnings
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.simplefilter(action='ignore', category=Warning)

##################################################
# Helper Functions
##################################################

# Data Preprocessing & Feature Engineering
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        Numerik fakat kategorik olan değişkenler için sınıf eşik değeri.
    car_th: int, float
        Kategorik fakat kardinal değişkenler için sınıf eşik değeri.

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi.
    num_cols: list
        Numerik değişken listesi.
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi.

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat, cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["object", "category", "bool", "uint8"]]

    num_but_cats = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"] and dataframe[col].nunique() < cat_th]

    cat_but_cars = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["object", "category"] and dataframe[col].nunique() > car_th]

    cat_cols = cat_cols + num_but_cats
    cat_cols = [col for col in cat_cols if col not in cat_but_cars]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_cars)}")
    print(f"num_but_cat: {len(num_but_cats)}")

    return cat_cols, num_cols, cat_but_cars


def outlier_thresholds(dataframe, num_col, q1=0.25, q3=0.75):
    quartile1 = dataframe[num_col].quantile(q1)
    quartile3 = dataframe[num_col].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr

    # outliers = [dataframe[(dataframe[num_col] < low) | (dataframe[num_col] > up)]]
    return low_limit, up_limit


def replace_with_thresholds(dataframe, num_col):
    low_limit, up_limit = outlier_thresholds(dataframe, num_col)

    dataframe.loc[(dataframe[num_col] < low_limit), num_col] = low_limit
    dataframe.loc[(dataframe[num_col] > up_limit), num_col] = up_limit
    
    
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    # fonksiyona girilen dataframe içerisinden kategorik değişkenleri one hot encoding işlemine sokuyoruz
    # sonrasında ana dataframe'e bunu kaydederek fonksiyon sonunda bu df'i döndürüyoruz.
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def diabetes_data_prep(df):
    df.columns = [col.upper() for col in df.columns]
    
    # Glucose
    df["NEW_GLUCOSE_CAT"] = pd.cut(x=df['GLUCOSE'], bins=[-1, 139, 200], labels=["normal", "prediabetes"])
    
    # Age
    df.loc[(df["AGE"] < 35), "NEW_AGE_CAT"] = 'young'
    df.loc[(df["AGE"] >= 35) & (df["AGE"] <= 55), "NEW_AGE_CAT"] = 'middleage'
    df.loc[(df["AGE"] > 55), "NEW_AGE_CAT"] = 'old'
    
    # BMI
    df["NEW_BMI_RANGE"] = pd.cut(x=df["BMI"], bins=[-1, 18.5, 24.9, 29.9, 100], labels=["underweight", "healty", "overweight", "obese"])
    
    # BloodPressure
    df["NEW_BLOODPRESSURE"] = pd.cut(x=df["BLOODPRESSURE"], bins=[-1, 79, 89, 123], labels=["normal", "hs1", "hs2"])
    
    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
    
    cat_cols = [col for col in cat_cols if col != "OUTCOME"]
    
    df = one_hot_encoder(df, cat_cols)
    
    df.columns = [col.upper() for col in df.columns]

    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

    cat_cols = [col for col in cat_cols if col != "OUTCOME"]

    replace_with_thresholds(df, "INSULIN")
    
    X_scaled = StandardScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)
    
    y = df["OUTCOME"]
    X = df.drop(["OUTCOME"], axis=1)
    
    return X, y


# Base Models
def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                  ('KNN', KNeighborsClassifier()),
                  ('SVC', SVC()),
                  ('CART', DecisionTreeClassifier()),
                  ('RF', RandomForestClassifier()),
                  ('Adaboost', AdaBoostClassifier()),
                  ('GBM', GradientBoostingClassifier()),
                  ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                  ('LightGBM', LGBMClassifier()),
                  # ('CatBoost', CatBoostClassifier(verbose=True))
                  ]
                   
    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name})")
        
        
# Hyperparameter Optimization
knn_params = {"n_neighbors": range(2,20)}

cart_params = {"max_depth": range(1, 20),
              "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
            "max_features": [5, 7, "auto"],
            "min_samples_split": [15, 20],
            "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                 "max_depth": [5, 8],
                 "n_estimators": [100, 200],
                 "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                  "n_estimators": [300, 500],
                  "colsample_bytree": [0.7, 1]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
              ('CART', DecisionTreeClassifier(), cart_params),
              ('RF', RandomForestClassifier(), rf_params),
              ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
              ('LightGBM', LGBMClassifier(), lightgbm_params)]

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")
        
        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)
        
        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


# Stacking & Ensemble Learning
def voting_classifier(best_models, X, y):
    print("Voting Classifier....")
    
    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
                                             ('RF', best_models["RF"]),
                                             ('LightGBM', best_models["LightGBM"])],
                                 voting='soft').fit(X, y)
    
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1 Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf


##################################################
# Pipeline Main Function
##################################################

def main():
    df = pd.read_csv("C:/Users/btskd/miuul_machine_learning/machine_learning/datasets/diabetes.csv")
    X, y = diabetes_data_prep(df)
    base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    voting_clf = voting_classifier(best_models, X, y)
    joblib.dump(voting_clf, "voting_clf.plk")
    return voting_clf


if __name__ == "__main__":
    main()