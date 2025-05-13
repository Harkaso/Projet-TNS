import json
import re
import sys
import traceback
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from mealpy import Problem
from mealpy.math_based.AOA import OriginalAOA
from mealpy.utils.space import FloatVar, IntegerVar
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

"""
To start please change directory to the current project directory: > cd {your_path}/analyseur/
then run the command: > streamlit run ./app.py
"""

def app(aoa_epoch=5, model_epoch=5):
    class Capturing(list):
        def __enter__(self):
            self._stdout = sys.stdout
            sys.stdout = self._stringio = StringIO()
            return self

        def __exit__(self, *args):
            self.extend(self._stringio.getvalue().splitlines())
            del self._stringio
            sys.stdout = self._stdout

    def result():
        if final_model is not None:
            final_model.save("./saves/best_model.keras")
            st.subheader("🎯 Résultats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{acc:2.2f}%")
            with col2:
                st.metric("Precision", f"{prec:2.2f}%")

            st.subheader("📉 Matrice de confusion")
            fig, ax = plt.subplots(figsize=(10, 8))
            labels = ["get down", "get up", "lying", "no_person", "sitting", "standing", "walking"]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel("Prédit")
            ax.set_ylabel("Réel")
            ax.set_title("Matrice de confusion")
            st.pyplot(fig)

            st.subheader("📈 Courbes d'apprentissage")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(history.history['accuracy'], label='Train Accuracy')
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Accuracy')
            ax1.legend()

            ax2.plot(history.history['loss'], label='Train Loss')
            ax2.plot(history.history['val_loss'], label='Validation Loss')
            ax2.set_title('Loss')
            ax2.legend()

            st.pyplot(fig)
        else:
            st.error("Le modèle final n'a pas pu être construit")

    def build_model(X_train, y_train, X_val, y_val, timesteps, features, y_dim, params):
        try:
            model = Sequential()
            model.add(Input(shape=(timesteps, features)))

            kernel_size = max(1, int(params[1]))
            model.add(Conv1D(
                filters=int(params[0]),
                kernel_size=kernel_size,
                activation='relu',
                padding='same'
            ))
            model.add(Dropout(float(params[3])))

            for _ in range(max(1, int(params[4]))):
                model.add(LSTM(50, return_sequences=True))
                model.add(Dropout(float(params[3])))

            model.add(Flatten())
            model.add(Dense(y_dim, activation='softmax'))

            model.compile(
                optimizer=Adam(learning_rate=float(params[2])),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=model_epoch,
                batch_size=max(16, int(params[5])),
                verbose=0
            )

            preds = model.predict(X_val)
            y_true = np.argmax(y_val, axis=1)
            y_pred = np.argmax(preds, axis=1)

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average="macro")

            labels = ["get down", "get up", "lying", "no_person", "sitting", "standing", "walking"]
            cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))

            return model, 1.0 - acc, acc, prec, cm, y_true, y_pred, history

        except Exception as e:
            st.error(f"Error in model building: {str(e)}")
            return None, 1.0, 0.0, 0.0, None, None, None, None

    def fitness(solution):
        try:
            if solution[1] < 1:
                return [1.0]
            if solution[5] < 16:
                return [1.0]

            _, error, _, _, _, _, _, _ = build_model(
                X_train, y_train, X_val, y_val,
                timesteps, features, y_dim, solution
            )
            return [error]
        except Exception as e:
            st.warning(f"Erreur dans fitness: {str(e)}")
            return [1.0]

    st.set_page_config(page_title="Optimisation AOA avec CNN-LSTM", layout="wide")
    st.title("📈 Optimisation AOA avec modèle CNN-LSTM")

    uploaded_file = st.file_uploader("📁 Charger un fichier CSV", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("📄 Aperçu du CSV")
        st.dataframe(data.head())

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.subheader("📊 Courbes des colonnes numériques")
            fig, ax = plt.subplots()
            for col in numeric_cols:
                ax.plot(data[col], label=col)
            ax.grid(True)
            st.pyplot(fig)

        try:
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            timesteps = 10
            features = X.shape[1] // timesteps
            X = X[:, :timesteps * features].reshape(-1, timesteps, features)

            le = LabelEncoder()
            y_encoded = to_categorical(le.fit_transform(y))
            y_dim = y_encoded.shape[1]

            X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2)

            st.success(f"✅ Données préparées. Forme X_train: {X_train.shape}, y_train: {y_train.shape}")

            use_aoa = st.checkbox("Utiliser l'optimisation AOA", value=True)

            if use_aoa:
                if st.button("🚀 Lancer l'optimisation AOA"):
                    with st.spinner("Optimisation AOA en cours..."):
                        bounds = [
                            IntegerVar(16, 128, name="num_filters"),
                            IntegerVar(1, 5, name="filter_size"),
                            FloatVar(1e-4, 1e-2, name="learning_rate"),
                            FloatVar(0.0, 0.5, name="dropout"),
                            IntegerVar(1, 3, name="num_lstm_layers"),
                            IntegerVar(16, 128, name="batch_size"),
                        ]

                        problem = Problem(
                            obj_func=fitness,
                            bounds=bounds,
                            minmax="min",
                            verbose=False
                        )

                        console_output = st.empty()
                        console_output.text("Début de l'optimisation AOA...")

                        with Capturing() as output:
                            model = OriginalAOA(epoch=aoa_epoch, pop_size=10)
                            g_best = model.solve(problem)

                        st.subheader("📝 Journal de la console AOA")
                        with st.expander("Voir les détails de l'optimisation"):
                            for line in output:
                                cleaned_line = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', line)
                                st.text(cleaned_line)

                        best_params = g_best.solution

                        st.subheader("✅ Meilleurs hyperparamètres AOA")
                        param_names = [
                            "num_filters", "filter_size", "learning_rate",
                            "dropout", "num_lstm_layers", "batch_size"
                        ]
                        best_params = [
                            int(best_params[0]),                    # num_filters
                            int(best_params[1]),                    # filter_size
                            round(np.float64(best_params[2]), 5),   # learning_rate
                            round(np.float64(best_params[3]), 2),   # dropout
                            int(best_params[4]),                    # num_lstm_layers
                            int(best_params[5])                     # batch_size
                        ]
                        best_params_dict = {name: val for name, val in zip(param_names, best_params)}
                        st.json(best_params_dict)
                        with open("saves/best_params.json", "w") as f:
                            json.dump(best_params_dict, f, indent=4)

                        with st.spinner("Construction du modèle final..."):
                            with Capturing() as model_output:
                                final_model, _, acc, prec, cm, y_true, y_pred, history = build_model(
                                    X_train, y_train, X_val, y_val,
                                    timesteps, features, y_dim, best_params
                                )

                            st.subheader("📝 Journal de la console (Modèle final)")
                            with st.expander("Voir les détails de la construction du modèle"):
                                for line in model_output:
                                    st.text(line)
                        result()
            else:
                st.subheader("🔧 Paramètres manuels")
                col1, col2 = st.columns(2)
                with col1:
                    num_filters = st.slider("Nombre de filtres", 16, 128, 64)
                    filter_size = st.slider("Taille du filtre", 1, 5, 3)
                    learning_rate = st.slider("Taux d'apprentissage", 5e-5, 1e-2, 1.5e-3, format="%.5f", step=5e-5)
                with col2:
                    dropout = st.slider("Dropout", 0.0, 0.5, 0.2)
                    num_lstm_layers = st.slider("Nombre de couches LSTM", 1, 3, 1)
                    batch_size = st.slider("Taille du batch", 16, 128, 32)

                if st.button("🏃‍♂️ Lancer l'entraînement"):
                    manual_params = [
                        num_filters, filter_size, learning_rate,
                        dropout, num_lstm_layers, batch_size
                    ]

                    with st.spinner("Entraînement en cours..."):
                        console_output = st.empty()
                        console_output.text("Début de l'entraînement...")

                        with Capturing() as output:
                            final_model, _, acc, prec, cm, y_true, y_pred, history = build_model(
                                X_train, y_train, X_val, y_val,
                                timesteps, features, y_dim, manual_params
                            )

                        st.subheader("📝 Journal de la console")
                        with st.expander("Voir les détails de l'entraînement"):
                            for line in output:
                                cleaned_line = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', line)
                                st.text(cleaned_line)
                    result()

        except Exception as e:
            st.error(f"Erreur pendant le traitement : {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")


if __name__ == '__main__':
     print("""
        To start please change directory to the current project directory: > cd {your_path}/analyseur/
        then run the command: > streamlit run ./app.py
            """)
     app()
