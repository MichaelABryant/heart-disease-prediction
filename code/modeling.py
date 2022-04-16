"""Modeling for UCI heart disease dataset.

Organization:
    main()
        get_preprocessed_data()
        get_model_baselines_and_CV_scores()
            get_models()
            get_cv_scores()
        get_model_evaluations()"""

# Import libraries.
import pandas as pd
import numpy as np
from numpy import mean, std
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pickle
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import (accuracy_score, recall_score, confusion_matrix,
                             classification_report, roc_curve, auc)
import warnings

# Suppress warnings.
warnings.filterwarnings("ignore")

def main():
    """Main function:
        1. Preprocesses UCI heart disease dataset,
        2. Calculates and prints model baselines and cross-validation scores,
        3. Calculates and prints test set accuracies and recall, and
        4. Calculates and prints confusion matrix, classification report,
            ROC, and ROC AUCs for the best classifier (LogisticRegression).
    Only executes when __name__ == "__main__"."""
    
    # Preprocess data.
    X_train, X_test, y_train, y_test = get_preprocessed_data()
    
    # Model selection.
    estimators = get_model_baselines_and_CV_scores(X_train, y_train)
    
    # Evaluate models.
    get_model_evaluations(estimators, X_train, y_train, X_test, y_test)

def get_preprocessed_data(drop_first=False):
    """Preprocesses dataframe by seperating target, creating dummy variables,
    train/test split, and scaling numerical variables. The scaler is
    pickled for use in deployment."""
        
    # Load data.
    df = pd.read_csv('../datasets/heart.csv')
    
    # There are two missing values in thal (where thal = 0), replace with NaN.
    df.loc[:,'thal'][df['thal'] == 0] = np.nan
    
    # Copy dataframe to X, assign target to y, and remove target from X
    X = df.copy()
    y = df.pop('target')
    
    # Save column names from predictor dataframe.
    column_names = X.columns
    
    # Create an instance of KNNImputer.
    knn_impute = KNNImputer(n_neighbors=20)
    
    # Fit and transform predictor dataframe with KNNImputer.
    X = knn_impute.fit_transform(X)
    
    # Convert numpy matrix to dataframe.
    X = pd.DataFrame(X)
    
    # Re-assign column names.
    X.columns = column_names
    
    # Define array with numerical feature names.
    numerical = [
        'age',
        'trestbps',
        'chol',
        'thalach',
        'oldpeak',
    ]

    # Define array with categorical feature names.
    categorical = [
        'sex',
        'cp',
        'fbs',
        'restecg',
        'exang',
        'slope',
        'ca',
        'thal'
    ]
    
    # Change dtype of categorical features to category.
    X[categorical] = X[categorical].astype('category')
    
    
    # Get dummies for categorical features and concatenate with numerical
    # features
    if drop_first == False:
        
        # Create an instance of OneHotEncoder.
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
        # Create dummy variables.
        cat_feature_names = X[categorical].columns
        X_cat = pd.DataFrame(enc.fit_transform(X[categorical]))
        X_cat.columns = enc.get_feature_names(cat_feature_names)
        
        # Concatenate numerical and categorical dummy dataframes.
        X = pd.concat([X[numerical], X_cat], axis=1)
        
        # # OneHotEncoder creates column names with both int and str dtypes.
        # X.columns = X.columns.astype(str)
        
        # Pickle fitted OneHotEncoder for deployment.
        outfile = open('../deployment/onehotencoder.pkl', 'wb')
        pickle.dump(enc,outfile)
        outfile.close()
        
    else:
        
        # Create an instance of OneHotEncoder with drop first.
        enc_lr = OneHotEncoder(handle_unknown='ignore', sparse=False,
                               drop='first')
        
        # Create dummy variables.
        cat_feature_names = X[categorical].columns
        X_cat = pd.DataFrame(enc_lr.fit_transform(X[categorical]))
        X_cat.columns = enc_lr.get_feature_names(cat_feature_names)
        
        # Create dummy variables for LogisticRegression inference.
        X = pd.concat([X[numerical], X_cat], axis=1)
        column_names = X.columns

    # Train/test split with stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        train_size=0.75,
                                                        random_state=1)

    # Create an instance of MinMaxScaler.
    scaler = MinMaxScaler()

    # Fit and transform numerical training data with MinMaxScaler.
    X_train[numerical] = scaler.fit_transform(X_train[numerical])
    
    # Transform numerical test data with MinMaxScaler.
    X_test[numerical] = scaler.transform(X_test[numerical])

    # Pickle fitted MinMaxScaler for deployment.
    outfile = open('../deployment/minmaxscaler.pkl', 'wb')
    pickle.dump(scaler,outfile)
    outfile.close()
    
    # When the preprocess function is used for general model training,
    # drop_first is set to False.
    if drop_first == False:
        return X_train, X_test, y_train, y_test
    
    # When the preprocess function is used for LogisticRegression inference,
    # drop_first is set to True.
    else:
        return X_train, y_train, column_names
    
    
def get_model_baselines_and_CV_scores(X_train, y_train):
    """Calculate and display classifier baselines and 10-Fold CV scores."""
    
    # Initialize models with default parameters.
    classifier_instances, classifier_names = get_models()
    
    # Retrieve model baselines.
    print("Baseline Mean Accuracy/Recall 10-Fold CV Results:")
    get_cv_scores(classifier_instances, X_train, y_train,
                  classifier_names)
    
    # Define parameter grids for GridSearchCV:
    # GaussianNB,   
    pg_gnb = {
        'var_smoothing': np.logspace(0,-10, num=100)
        }
    # LogisticRegression,
    pg_lr = {
        'C' : np.arange(.5,1.5,.1)
        }
    # DecisionTreeClassifier,
    pg_dt = {
        'criterion':['gini','entropy'],
        'splitter': ['best', 'random'],
        'max_depth': np.arange(1, 15, 1),
        }
    # KNeighborsClassifier,
    pg_knn = {
        'n_neighbors' : np.arange(15,20,1),
        'weights' : ['uniform', 'distance'],
        'algorithm' : ['auto', 'ball_tree','kd_tree','brute'],
        'p' : [1,2,3,4,5]
        }
    # RandomForestClassifier,
    pg_rf =  {
        'n_estimators': [500], 
        'bootstrap': [True,False],
        'max_features': ['auto','sqrt'],
        }
    # SVC,
    pg_svc = {
        'kernel': ['linear', 'poly', 'sigmoid','rbf'],
        'gamma': [1e-3, 1e-4, 1e-5, 1e-6],
        'C': np.arange(50,75,1)
        }
    # and XGBClassifier.
    pg_xgb = {
        'max_depth': np.arange(2,15,1),
        'n_estimators': np.arange(20,80,10),
        'learning_rate': np.arange(.1,1.5,.1)
        }
    
    # Combine parameter grids into a list.
    parameter_grids = [pg_gnb, pg_lr, pg_dt, pg_knn, pg_rf, pg_svc, pg_xgb]
    
    # Initialize list for best gridsearch parameters.
    best_parameters = list()
    
    # Perform GridSearchCV to find best parameters.
    print("\nGridSearchCV Mean Accuracy/Recall 10-Fold CV Results:")
    for idx, classifier_instance in enumerate(classifier_instances):
    
        # Perform GridSearchCV with accuracy and recall as scoring.
        gs = GridSearchCV(classifier_instance, param_grid=parameter_grids[idx],
                          scoring=["accuracy","recall"], refit="accuracy",
                          cv=10, n_jobs=-1)
        best_gs = gs.fit(X_train,y_train)
        print('{} | accuracy: {} +/- {}'.format(
            classifier_names[idx],
            str(round(best_gs.cv_results_['mean_test_accuracy'][-1],2)),
            str(round(best_gs.cv_results_['std_test_accuracy'][-1],2))),
            " | recall: {} +/- {} ".format(
                str(round(best_gs.cv_results_['mean_test_recall'][-1],2)),
                str(round(best_gs.cv_results_['std_test_recall'][-1],2))))
        
        # Append best parameters from GridSearchCV to best_parameters list.
        best_parameters.append(best_gs.best_params_)
    
    # Initialize classifiers with best parameters from GridSearchCV.
    classifier_instances, classifier_names = get_models(best_parameters)
    
    # Initialize ensemble instances using classifiers with best parameters.
    # Define base classifiers.
    level0 = list()
    for idx, classifier_instance in enumerate(classifier_instances):
        level0.append((classifier_names[idx], classifier_instance))

    # Define meta learner model.
    level1 = LogisticRegression()
    
    # Create stacking classifier instance.
    stacking = StackingClassifier(estimators=level0, final_estimator=level1,
                                  cv=10)
    
    # Create soft voting classifier instance.
    soft_voting = VotingClassifier(estimators=level0, voting='soft')
    
    # Create hard voting classifier instance.
    hard_voting = VotingClassifier(estimators=level0, voting='hard')
    
    # Combine ensemble instances and names in lists.
    ensemble_instances = [stacking, soft_voting, hard_voting]
    ensemble_names = ["stk", "sv", "hv"]
    
    # Perform 10-Fold CV for ensemble estimators.
    print("\nEnsemble Mean Accuracy/Recall 10-Fold CV Results:")
    get_cv_scores(ensemble_instances, X_train, y_train, ensemble_names)
    
    # Combine lists of all estimators and their names.    
    all_estimators = classifier_instances + ensemble_instances
    all_estimator_names = classifier_names + ensemble_names
    
    return [all_estimators, all_estimator_names]


def get_models(best_params = dict()):
    """Create instances of models with the option of passing the best
    parameters."""
    
    # Create instances of models with default parameters.
    gnb = GaussianNB()
    lr = LogisticRegression(max_iter = 2000)
    dt = tree.DecisionTreeClassifier(random_state = 1)
    knn = KNeighborsClassifier()
    rf = RandomForestClassifier(random_state = 1)
    svc = SVC(probability = True)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='error',
                        random_state =1)
    
    # Create lists with estimator instances and names.
    classifier_instances = [gnb, lr, dt, knn, rf, svc, xgb]
    classifier_names = ["gnb", "lr", "dt", "knn", "rf", "svc", "xgb"]
    
    # Set parameters if best_params is passed with get_models.
    if best_params != dict():
        
        for idx, classifier_instance in enumerate(classifier_instances):
            
            classifier_instance.set_params(**best_params[idx])
            
    return classifier_instances, classifier_names


def get_cv_scores(classifier_instances, X_train, y_train, classifier_names,
                  cv_folds=10):
    """Retrieve model baseline. Input is instance of model."""
    
    for idx, classifier_instance in enumerate(classifier_instances):
        # Perform 10-fold CV with accuracy and recall as scoring.
        cv = cross_validate(classifier_instance, X_train, y_train,
                            scoring=("accuracy","recall"), cv=cv_folds)
        print("{} | accuracy: ".format(classifier_names[idx]),
              round(mean(cv['test_accuracy']),2), '+/-',
              round(std(cv['test_accuracy']),2), " | recall: ",
              round(mean(cv['test_recall']),2), '+/-',
              round(std(cv['test_recall']),2))


def get_model_evaluations(estimators, X_train, y_train, X_test, y_test):
    """Calculate and display test set accuracy and recall for all classifiers.
    Calculate and display confusion matrix, classification report, ROC, and
    ROC AUC for best model (LogisticRegression)."""
    
    # Unpack estimators list.
    estimators, estimator_names = estimators[0], estimators[1]
    
    # Calculate accuracy and recall scores for test set for all estimators.    
    print("\nTest Accuracy/Recall Results:")
    for idx, estimator in enumerate(estimators):
        # Fit estimator to all training data.
        estimator = estimator.fit(X_train, y_train)
        
        # Predict on test set with fitted estimator.
        y_pred = estimator.predict(X_test)
        print('{} | accuracy: {} | recall: {}'.format(
            estimator_names[idx],round(accuracy_score(y_test, y_pred),2),
            round(recall_score(y_test, y_pred),2)))
        
    # Unpack best classifier (LogisticRegression).
    lr_clf= estimators[1]
    
    # LogisticRegression final fit and prediction.
    lr_clf.fit(X_train,y_train)
    y_pred_lr = lr_clf.predict(X_test)
    
    # Pickle LogisticRegression model.
    outfile = open('../deployment/logisticregression.pkl', 'wb')
    pickle.dump(lr_clf,outfile)
    outfile.close()
    
    # Get dummy variables, but drop first and fit LogisticRegression
    X_train, y_train, column_names = get_preprocessed_data(drop_first=True)
    lr_clf_drop_first = LogisticRegression(C=0.5)
    lr_clf_drop_first.fit(X_train, y_train)    
    
    # Plot a LogisticRegression confusion matrix.
    matrix_lr = confusion_matrix(y_test, y_pred_lr)
    plt.figure(figsize=(16,7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix_lr, annot=True, annot_kws={'size':10}, linewidths=0.2)
    plt.xticks()
    plt.yticks()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion matrix for LogisticRegression')
    plt.savefig('../output/modeling/confusion_matrix_logistic_regression.jpg',
                bbox_inches='tight')
    plt.show()

    print('\nLogisticRegression Metrics\n')

    # Display the LogisticRegression classification report.
    print(classification_report(y_test, y_pred_lr))
    
    
    # Plot an ROC for LogisticRegression.
    pred_prob_lr = lr_clf.predict_proba(X_test)
    fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, pred_prob_lr[:,1], 
                                              pos_label=1)
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(fpr_lr, tpr_lr, label='LogisticRegression')
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.title('ROC for LogisticRegression (AUC: 0.94)')
    plt.xlabel('False positive rate (1 - specificity)')
    plt.ylabel('True positive rate (sensitivity)')
    plt.grid(True)
    plt.savefig('../output/modeling/ROC_logisticregression.jpg',
                bbox_inches='tight')
    plt.show()
    
    # Display ROC AUC scores for LogistictRegression.
    print('LogisticRegression AUC (ROC): {}'.format(auc(fpr_lr, tpr_lr)))
    
    # Create sorted dataframe with LogisticRegression coefficients.
    lr_coefficients = pd.DataFrame([lr_clf_drop_first.coef_[0]],
                                   columns = column_names)
    sorted_lr = lr_coefficients.iloc[:, np.argsort(lr_coefficients.loc[0])]

    plt.figure(figsize=(16,7))
    sns.barplot(y=sorted_lr.columns, x=sorted_lr.iloc[0,:])
    plt.xlabel('LogisticRegression coefficients')
    plt.savefig('../output/modeling/logistic_regression_coefficients.jpg',
                bbox_inches='tight')
    plt.show()        

if __name__ == "__main__":
    main()
