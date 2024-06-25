import streamlit as st
import pickle
from sklearn.feature_extraction import TfidfVectorizer

# Load the pickle models
model_nb = pickle.load(open('nb_model.pkl', 'rb'))
model_rf = pickle.load(open('rf_model.pkl', 'rb'))
model_svm = pickle.load(open('lr_model.pkl', 'rb'))  # Assuming lr_model.pkl is actually for SVM

# Load the TF-IDF vectorizer
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Function to predict gender
def predict_gender(name, model, vectorizer):
    name_vectorized = vectorizer.transform([name])
    prediction = model.predict(name_vectorized)
    confidence = None
    
    if hasattr(model, 'predict_proba'):  # Check if the model has predict_proba() method
        confidence = model.predict_proba(name_vectorized)
    
    return prediction[0], confidence

# Streamlit app
def main():
    st.title('Prediksi Jenis Kelamin Berdasarkan Nama')
    st.write('Masukkan nama untuk diprediksi jenis kelaminnya.')

    name = st.text_input('Masukkan nama:')
    
    if st.button('Prediksi'):
        # Predict using each model
        pred_nb, confidence_nb = predict_gender(name, model_nb, tfidf_vectorizer)
        pred_rf, confidence_rf = predict_gender(name, model_rf, tfidf_vectorizer)
        pred_svm, confidence_svm = predict_gender(name, model_svm, tfidf_vectorizer)
        
        # Convert predictions to gender labels
        pred_nb_label = 'Pria' if pred_nb == 1 else 'Wanita'
        pred_rf_label = 'Pria' if pred_rf == 1 else 'Wanita'
        pred_svm_label = 'Pria' if pred_svm == 1 else 'Wanita'
        
        # Display predictions
        st.subheader('Hasil Prediksi')
        st.write(f'Nama: {name}')
        st.write('Prediksi Naive Bayes:', pred_nb_label)
        st.write('Prediksi Random Forest:', pred_rf_label)
        st.write('Prediksi Logistic Regression:', pred_svm_label)
        
        # Display confidence scores (if available)
        st.subheader('Confidence Score (Jika Tersedia)')
        if confidence_nb is not None:
            st.write(f'Confidence Naive Bayes: {confidence_nb[0][pred_nb]:.2%}')
        else:
            st.write('Confidence Naive Bayes: Tidak tersedia')
        
        if confidence_rf is not None:
            st.write(f'Confidence Random Forest: {confidence_rf[0][pred_rf]:.2%}')
        else:
            st.write('Confidence Random Forest: Tidak tersedia')
        
        if confidence_svm is not None:
            st.write(f'Confidence Logistic Regression: {confidence_svm[0][pred_svm]:.2%}')
        else:
            st.write('Confidence Logistic Regression: Tidak tersedia')

if __name__ == '__main__':
    main()
