

#import libraries for randon forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data
df_train = pd.read_parquet('train_df.parquet', engine='pyarrow')
df_test = pd.read_parquet('test_df.parquet', engine='pyarrow')

# Separate features X and target y (anomaly label 0 or 1)
X_train = df_train.drop('is_anomalous', axis=1)
y_train = df_train['is_anomalous']

X_test = df_test.drop('is_anomalous', axis=1)
y_test = df_test['is_anomalous']

# Create a Random Forest Classifier model 
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model ### NEED TO FIX HERE AND BELOW
#model.fit(X_train, y_train)

# Make predictions on the test set
#y_pred = model.predict(X_test)

# Evaluate the model
#accuracy = accuracy_score(y_test, y_pred)
#print(f"Accuracy: {accuracy}")

