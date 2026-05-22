import pickle
from sklearn.metrics import accuracy_score
import numpy as np

# load your saved model
model = pickle.load(open('model.pkl', 'rb'))  # change filename to whatever yours is

# load your test data
# replace these with however you load your data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")