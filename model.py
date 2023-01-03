import os
# import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense





methods = ['RF', 'DT', 'SVM', 'NB', 'NN']
typeOfModel = methods[1]

# read dataframe
dataset_path = '/home/shabnam/pouya/mainDatasets/merged_dataset.csv'
df = pd.read_csv(dataset_path)
df = df.iloc[:,1:]

# shuffling
df = df.sample(frac=1).reset_index(drop=True)
# df = df.iloc[1:,]



# create X and y
X = df.iloc[:,1:-1]
y = df.iloc[:,-1]
print(X.head())

#train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state=42) # 80% training and 20% test

print(len(X_train))
print(len(X_test))


################## DT Classifier
DT = DecisionTreeClassifier(criterion="entropy")
DT_model = DT.fit(X=X_train, y=y_train)

# test
y_pred = DT_model.predict(X_test)
print(f"""Performance of our DecisionTree (DT) model: 
      \t Accuracy = {accuracy_score(y_test, y_pred)} 
      \t Precision = {precision_score(y_test,y_pred)} 
      \t Recall = {recall_score(y_test, y_pred)} 
      \t F1 = {f1_score(y_test, y_pred)}""")



################## NB Classifier
gnb = GaussianNB()
NB_model = gnb.fit(X=X_train, y=y_train)

# test
y_pred = NB_model.predict(X_test)
print(f"""Performance of our Naive Bays (NB) model: 
      \t Accuracy = {accuracy_score(y_test, y_pred)} 
      \t Precision = {precision_score(y_test,y_pred)} 
      \t Recall = {recall_score(y_test, y_pred)} 
      \t F1 = {f1_score(y_test, y_pred)}""")


################## SVM Classifier
SVM = svm.SVC()
SVM_model = SVM.fit(X=X_train, y=y_train)

# test
y_pred = SVM_model.predict(X_test)
print(f"""Performance of our Support Vector Machine (SVM) model: 
      \t Accuracy = {accuracy_score(y_test, y_pred)} 
      \t Precision = {precision_score(y_test,y_pred)} 
      \t Recall = {recall_score(y_test, y_pred)} 
      \t F1 = {f1_score(y_test, y_pred)}""")

################## RF Classifier
RF = RandomForestClassifier(random_state=0) #max_depth=2,
RF_model = RF.fit(X=X_train, y=y_train)

# test
y_pred = RF_model.predict(X_test)
print(f"""Performance of our Random Forest (RF) model: 
      \t Accuracy = {accuracy_score(y_test, y_pred)} 
      \t Precision = {precision_score(y_test,y_pred)} 
      \t Recall = {recall_score(y_test, y_pred)} 
      \t F1 = {f1_score(y_test, y_pred)}""")

################## NN Classifier

# define the keras model
model = Sequential()
model.add(Dense(4, input_shape=(768,), activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
history = model.fit(X_train, y_train, epochs=5, batch_size=200, validation_data=(X_test, y_test))


# evaluate the keras model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

# test
y_pred = model.predict(X_test)
# y_pred = model.predict_classes(testX, verbose=0)
threshold = 0.50
new_y_predict = []
for i in y_pred:
    if i >= threshold:
        new_y_predict.append(1)
    else:
        new_y_predict.append(0)
y_pred = new_y_predict

# print(y_pred)
# print(y_test)


print(f"""Performance of our Neural Network (NN) model: 
      \t Accuracy = {accuracy_score(y_test, y_pred)} 
      \t Precision = {precision_score(y_test,y_pred)} 
      \t Recall = {recall_score(y_test, y_pred)} 
      \t F1 = {f1_score(y_test, y_pred)}""")



### Plot
# evaluate the model
from matplotlib import pyplot
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig('a.png')
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.savefig('b.png')







