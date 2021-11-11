##############################################################################
### William Brasic  ##########################################################
### Fatma Nasoz     ##########################################################
### CS 422/622      ##########################################################
### 5 November 2021 ##########################################################
##############################################################################


##############################################################################
############################## Homework 4 ####################################
##############################################################################


##############################################################################
### In this program we will use logistic regression to predict whether #######
###          a forest fire in Algeria does or does not happen.         #######
##############################################################################


## importing necessary libraries ##
import pandas as pd, os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, log_loss


## setting working directory ##
os.chdir(r'C:\Users\wbras\OneDrive\Desktop\UNLV\Fall_2021\CS_622\622_HW4')


## reading in data ##
df = pd.read_csv('622_HW4_Data.csv', header = 1)


##############################################################################
### DATA PREPROCESSING #######################################################
##############################################################################


## removing blank spaces ##
df.columns = df.columns.str.strip()
df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)


## converting the outcome to a binary variable ##
df.Classes = df.Classes.apply(lambda x: 0 if x == 'not fire' else 1)


## selecting dependent variable and covariates for the model ##
df = df['Classes Temperature RH Rain FWI DC'.split()]


## renaming columns ##
df.rename({'Classes': 'Class'}, axis=1, inplace=True)


## converting object datatypes to numeric datatypes ##
df.FWI = pd.to_numeric(df.FWI, errors = 'coerce')
df.DC = pd.to_numeric(df.DC, errors = 'coerce')


## checking the total NA values ##
df.isna().sum()


## filtering out data with no NA values ##
df[df.isnull().any(axis=1)]


## filling NA values with medians ##
df.FWI.median();
df.DC.median();
df.FWI.fillna(df.FWI.median(), inplace = True)
df.DC.fillna(df.DC.median(), inplace = True)
df.iloc[165,:];


## seperating the covariates from the outcome variable ##
df_Y, df_X = df.Class, df.iloc[:, 1:]


## split df into training and test dataframes ##
x_train, x_test, y_train, y_test = train_test_split(df_X, df_Y, test_size = 0.2, random_state = 622)


##############################################################################
### LOGISTIC REGRESSION MODEL ################################################
##############################################################################
    

## creating logistic regression model ##
lr_model = SGDClassifier(learning_rate = 'constant', eta0 = 0.001, loss = 'log', random_state = 622)


## fitting lr_model ##
lr_model.fit(x_train, y_train)


## predicting test and training data using lr_model ##
y_pred_test =  lr_model.predict(x_test).round(2) 
y_pred_train = lr_model.predict(x_train).round(2) 


## lr_model information ##
print('Model Information')
print('Coefficients:', lr_model.coef_.round(2))
print('Intercept:', lr_model.intercept_.round(2))
print('Number of Iterations until Convergence:', lr_model.n_iter_)
print('---------------------------------------------------------------------')


##############################################################################
### EVALUATION METRICS #######################################################
##############################################################################


## lr_model metrics for test set predictions ##
print('Evaluation Metrics for Test Set predictions')
print(f'True Negative: {confusion_matrix(y_test, y_pred_test)[0,0]}')
print(f'True Positive: {confusion_matrix(y_test, y_pred_test)[1,1]}')
print(f'False Negative: {confusion_matrix(y_test, y_pred_test)[1,0]}') 
print(f'False Positive: {confusion_matrix(y_test, y_pred_test)[0,1]}')
print(f'Accuracy: {accuracy_score(y_test, y_pred_test).round(2)}')
print(f'Sensitivity: {recall_score(y_test, y_pred_test).round(2)}')
print(f'Specificity: {confusion_matrix(y_test, y_pred_test)[0,0] / (confusion_matrix(y_test, y_pred_test)[0,0] + confusion_matrix(y_test, y_pred_test)[0,1])}')
print(f'F1 Score: {f1_score(y_test, y_pred_test).round(2)}')
print(f'Log Loss: {log_loss(y_test, y_pred_test).round(2)}')
print('---------------------------------------------------------------------')


## lr_model metrics for training set predictions ##
print('Evaluation Metrics for Training Set predictions')
print(f'True Negative: {confusion_matrix(y_train, y_pred_train)[0,0]}')
print(f'True Positive: {confusion_matrix(y_train, y_pred_train)[1,1]}')
print(f'False Negative: {confusion_matrix(y_train, y_pred_train)[1,0]}') 
print(f'False Positive: {confusion_matrix(y_train, y_pred_train)[0,1]}')
print(f'Accuracy: {accuracy_score(y_train, y_pred_train).round(2)}')
print(f'Sensitivity: {recall_score(y_train, y_pred_train).round(2)}')
print(f'Specificity: {(confusion_matrix(y_train, y_pred_train)[0,0] / (confusion_matrix(y_train, y_pred_train)[0,0] + confusion_matrix(y_train, y_pred_train)[0,1])).round(2)}')
print(f'F1 Score: {f1_score(y_train, y_pred_train).round(2)}')
print(f'Log Loss: {log_loss(y_train, y_pred_train).round(2)}')






    