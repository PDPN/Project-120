import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("diabetes.csv")
print(df.head())

X = df[["glucose","bloodpressure"]]
Y = df[["diabetes"]]
x_train_1,x_test_1,y_train_1,y_test_1 = train_test_split(X,Y,test_size=0.25,random_state=42)

sc = StandardScaler()

x_train_1 = sc.fit_transform(x_train_1)
x_test_1 = sc.fit_transform(x_test_1)

model_1 = GaussianNB()
model_1.fit(x_train_1, y_train_1)

y_pred_1 = model_1.predict(x_test_1)
accuracy = accuracy_score(y_test_1, y_pred_1)
print(accuracy)