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
    data.fillna(data.mean(), inplace=True)  # Mengisi missing values dengan nilai rata-rata
    data['sex'] = data['sex'].astype('category')
    data['cp'] = data['cp'].astype('category')
    # ...
    # Konversi fitur kategorikal menjadi numerik menggunakan LabelEncoder
    le = LabelEncoder()
    data['sex'] = le.fit_transform(data['sex'])
    data['cp'] = le.fit_transform(data['cp'])
    # ...
    
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    target = 'target'
    
    X = data[features]
    y = data[target]
    
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
    age = st.number_input("Usia:")
    sex = st.selectbox("Jenis Kelamin:", ["Laki-laki", "Perempuan"])
    cp = st.selectbox("Jenis Nyeri Dada:", ["Tipe 0", "Tipe 1", "Tipe 2", "Tipe 3"])
    # ...
    # Tambahkan input pengguna untuk fitur-fitur lainnya
    
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        # ...
        # Tambahkan fitur-fitur lainnya sesuai dengan input pengguna
    })
    
    # Preprocess input data
    input_data['sex'] = le.transform(input_data['sex'])
    input_data['cp'] = le.transform(input_data['cp'])
    # ...
    
    # Perform prediction
    prediction = predict(model, input_data)
    
    # Display the prediction result
    if prediction[0] == 0:
        st.write("Hasil: Tidak Terdiagnosis Hipertensi")
    else:
        st.write("Hasil: Terdiagnosis Hipertensi")

if __name__ == '__main__':
    main()
