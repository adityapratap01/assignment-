import pandas as pd

LoanData = pd.read_csv("loanstatus.csv")
LoanPrep = LoanData.copy()
LoanPrep.isnull().sum(axis=0)
LoanPrep = LoanPrep.dropna()
LoanPrep = LoanPrep.drop(['gender'], axis=1)
LoanPrep.dtypes
new = ['married','ch', 'status']
from sklearn.preprocessing import LabelEncoder
LoanPrep[new] = LoanPrep[new].astype('category')
LoanPrep.dtypes
label_encoder = LabelEncoder()
for column in new:
    LoanPrep[column] = label_encoder.fit_transform(LoanPrep[column])
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
LoanPrep['income'] = scalar.fit_transform(LoanPrep[['income']])
LoanPrep['loanamt'] = scalar.fit_transform(LoanPrep[['loanamt']])
X = LoanPrep.iloc[:, :-1]
Y = LoanPrep.iloc[:, -1]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
train_test_split(X, Y, test_size = 0.3, random_state = 1234, stratify=Y)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_predict = lr.predict(X_test)
Y_validate = lr.predict(X_train)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_predict)
score = lr.score(X_test, Y_test)
cm_validate = confusion_matrix(Y_train, Y_validate)
score = lr.score(X_test, Y_test)
score_validate = lr.score(X_train,Y_train)

