import joblib

model = joblib.load("bigFinal/pesistance_model.pkl")
new_data = [3, 1, 37, 2, 5, 11.56, True]
predictions = model.predict(new_data)

print(predictions)