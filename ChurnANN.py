import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import seaborn as sn

#Importing the dataset
df =pd.read_csv('Churn_Modelling.csv')

#Dropping useless in NN columns
df.drop(['CustomerId','Surname','RowNumber'],axis = 1 ,inplace = True)

#Replacing Gender values with 1,0
df.Gender.replace({"Female" : 1 ,"Male" : 0 } ,inplace=True)

#OneHotEncoding Geography Column
df1 = pd.get_dummies(data=df , columns = ['Geography'])
for col in df1:
    print(f'{col} : {df1[col].unique()}')

#Scaling the numbers
cols_to_scale = ['CreditScore' , 'Age' , "Tenure", "Balance" ,"NumOfProducts","EstimatedSalary"]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1[cols_to_scale] = scaler.fit_transform(df1[cols_to_scale])

#Acquiring X,y
X = df1.drop('Exited',axis=1)
y=df1['Exited']

#Splitting the Dataset into Train,Test
from sklearn.model_selection import train_test_split
X_train , X_test ,y_train , y_test = train_test_split(X,y,test_size = 0.2 , random_state = 5)

#Building and Compiling the model to train
model = keras.Sequential([
    keras.layers.Dense(20 , input_shape = (12,) ,activation = 'relu'),
    keras.layers.Dense(14 , activation='relu'),
    keras.layers.Dense(7 , activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])

#Training the model
model.fit(X_train,y_train,epochs=50)

#model evaluation
model.evaluate(X_test,y_test)

y_predicted = model.predict(X_test)

#Converting 2D to 1,0
y_pred = []
for element in y_predicted:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

#Classification Report
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,y_pred))

#Confusion Matrix
cm=tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize=(10,10))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel("Truth")