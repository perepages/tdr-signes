import pickle       # library to save dataset, modules, info...
import numpy as np  # library that helps to do mathematical operations and convert veriables into others

# (sk) is for speficially training the model
# here we import these special modules for the "training"
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data_dict = pickle.load(open('./FINAL/data.pickle', 'rb'))

# convert them into np.asarrays because they are lists in the other file
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# 2 sets (trainer and test the performance)
# here we separe the information in 2, the data and labels
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)


model = RandomForestClassifier() # simple arleady made algoryth for training
#training the classifier
model.fit(x_train, y_train)
# making the predictions (prediction means classifying the info but not giving any name yet)
y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly'.format(score*100))

# Save the model in a file
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
        