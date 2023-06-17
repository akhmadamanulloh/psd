import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache
def load_dataset():
    data = pd.read_csv('hypertension_data.csv')
    return data

# Preprocess dataset
def preprocess_data(data):
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    target = 'target'
    
    X = data[features]
    y = data[target]
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Train the model
def train_model(X_train, X_test, y_train):
    model = CategoricalNB()
    model.fit(X_train, y_train)
    return model

# Perform prediction
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Main function
def main():
    # Load dataset
    data = load_dataset()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Train the model
    model = train_model(X_train, X_test, y_train)
    
    # User input
    st.title("Deteksi Penyakit Hipertensi")
    age = st.number_input("Usia", min_value=0)
    sex = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    cp = st.selectbox("Jenis Nyeri Dada", [0, 1, 2, 3])
    trestbps = st.number_input("Tekanan Darah Istirahat", min_value=0)
    chol = st.number_input("Kolesterol", min_value=0)
    fbs = st.selectbox("Gula Darah Puasa", [0, 1])
    restecg = st.selectbox("EKG Istirahat", [0, 1, 2])
    thalach = st.number_input("Denyut Jantung Maksimal", min_value=0)
    exang = st.selectbox("Angina yang diinduksi olahraga", [0, 1])
    oldpeak = st.number_input("Depresi ST yang Dicetak oleh Latihan Relatif terhadap Istirahat", min_value=0)
    slope = st.selectbox("Kemiringan Segment ST", [0, 1, 2])
    ca = st.selectbox("Jumlah Pembuluh Utama yang Dicat", [0, 1, 2, 3])
    thal = st.selectbox("Hasil Tes Thalium", [0, 1, 2, 3])
    
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })
    
    # Perform prediction
    prediction = predict(model, input_data)
    
    # Display result
    st.subheader("Hasil Prediksi")
    if prediction[0] == 0:
        st.write("Tidak Terdiagnosis Hipertensi")
    else:
        st.write("Terdiagnosis Hipertensi")

if __name__ == '__main__':
    main()
