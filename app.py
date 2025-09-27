import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load and preprocess dataset
# -----------------------------
df = pd.read_csv("Titanic-Dataset.csv")

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop unused columns
df = df.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'])

# Encode categorical variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Scale numerical features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# -----------------------------
# Train model
# -----------------------------
X = df.drop(columns=['Survived'])
y = df['Survived']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚢 Titanic Survival Predictor")

# User inputs
age = st.slider("Age", 0, 80, 25)
fare = st.slider("Fare", 0, 500, 32)
sex_male = st.selectbox("Sex", ["Female", "Male"])
pclass = st.selectbox("Pclass", [1, 2, 3])
sibsp = st.slider("Siblings/Spouses aboard", 0, 8, 0)
parch = st.slider("Parents/Children aboard", 0, 6, 0)
embarked_Q = st.selectbox("Embarked Q?", ["No", "Yes"])
embarked_S = st.selectbox("Embarked S?", ["No", "Yes"])

# Prepare input data
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Sex_male': [1 if sex_male == "Male" else 0],
    'Embarked_Q': [1 if embarked_Q == "Yes" else 0],
    'Embarked_S': [1 if embarked_S == "Yes" else 0]
})

# Prediction
prediction = model.predict(input_data)

st.subheader("🧾 Prediction Result")
st.write("✅ Survived" if prediction[0] == 1 else "❌ Did not survive")
