
import pandas as pd
from ydata_profiling import ProfileReport

file_path = '/content/drive/MyDrive/Dataset/laptops.csv'
df = pd.read_csv(file_path)

from google.colab import drive
drive.mount('/content/drive')

prof_report = ProfileReport(df)

prof_report.to_notebook_iframe()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

datasetCSV = pd.read_csv('/content/drive/MyDrive/Dataset/laptops.csv')

X = datasetCSV[['Price', 'Rating']]
y = datasetCSV['year_of_warranty']

brand_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
brand_encoded = brand_encoder.fit_transform(datasetCSV[['brand']])

X = pd.concat([pd.DataFrame(brand_encoded), X], axis=1)

X.columns = X.columns.astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy (Random Forest):", accuracy)

scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)


svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)


new_instance = pd.DataFrame({
    'Price': [2000],
    'Rating': [4.5],
    'brand': ['asus']
})

new_instance_brand_encoded = brand_encoder.transform(new_instance[['brand']])

new_instance_encoded = pd.concat([pd.DataFrame(new_instance_brand_encoded), new_instance[['Price', 'Rating']]], axis=1)

new_instance_encoded.columns = new_instance_encoded.columns.astype(str)

new_instance_scaled = scaler.transform(new_instance_encoded)

prediction_rf = rf_model.predict(new_instance_scaled)

prediction_svm = svm_model.predict(new_instance_scaled)

if prediction_rf[0] == 1:
    print("Random Forest predicts the warranty to be of year 2 or more.")
else:
    print("Random Forest predicts the warranty to be of year less than 2.")

if prediction_svm[0] == 1:
    print("SVM predicts the warranty to be of year 2 or more.")
else:
    print("SVM predicts the warranty to be of year less than 2.")
