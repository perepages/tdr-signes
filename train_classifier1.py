# ALL THE FILES THAT HAVE THE NUMBER (1) BEHIND THEIR NAME ARE RELATED TO
# THE CODE FOR SIGNS WITH THE TWO HANDS, THATS WHY SOMETIMES THE CODE VARIES
# AND SO DO THE COMMENTS.


import pickle       # library to save dataset, modules, info...
import numpy as np  # library that helps to do mathematical operations and convert veriables into others

# (sk) is for speficially training the model
# here we import these special modules for the "training"
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
data_dict = pickle.load(open('./FINAL/data.pickle1', 'rb'))

# Convert them into lists because they might be of varying lengths
data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Determine the maximum length of sequences
max_length = max(len(seq) for seq in data)

# Pad sequences with zeros to ensure they all have the same length
padded_data = np.array([np.pad(seq, (0, max_length - len(seq))) if len(seq) < max_length else np.array(seq[:max_length]) for seq in data])

# Convert labels to a NumPy array
labels = np.asarray(labels)

# Check the shapes and types of data and labels
print(f"Data shape: {padded_data.shape}, Labels shape: {labels.shape}")
print(f"Data type: {type(padded_data)}, Labels type: {type(labels)}")

# Ensure the data is 2D and labels are 1D
if len(padded_data.shape) != 2:
    raise ValueError(f"Data should be 2D, but got shape: {padded_data.shape}")
if len(labels.shape) != 1:
    raise ValueError(f"Labels should be 1D, but got shape: {labels.shape}")

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(padded_data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the model
model = RandomForestClassifier()  # Simple algorithm for training
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly'.format(score * 100))

# Save the model to a file
with open('model.p1', 'wb') as f:
    pickle.dump({'model': model}, f)
