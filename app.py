import json
import joblib
import os
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Predicción Billboard", layout="centered")
st.title("Predicción de Éxito Billboard Top 1 - Eduin Zelaya - ISC")


@st.cache_resource
def cargar_modelo():
  pkl = (
      "/content/models/arbol_billboard.pkl"
      if os.path.exists("/content/models/arbol_billboard.pkl")
      else "arbol_billboard.pkl"
  )
  meta = (
      "/content/models/model_metadata.json"
      if os.path.exists("/content/models/model_metadata.json")
      else "model_metadata.json"
  )

  modelo = joblib.load(pkl)

  with open(meta, encoding="utf-8") as f:
    metadata = json.load(f)

  return modelo, metadata


@st.cache_data
def cargar_base():
  csv = (
      "/content/billboard.csv"
      if os.path.exists("/content/billboard.csv")
      else "billboard.csv"
  )
  return pd.read_csv(csv)


# Intentar cargar recursos
try:
  modelo, metadata = cargar_modelo()
  df = cargar_base()
  columnas_entrenamiento = metadata.get(
      "variables_modelo",
      ["artist_type", "mood", "tempo", "genre", "durationSeg"],
  )
except Exception as e:
  st.error(
      f"Error al cargar los archivos. Asegúrate de tener 'arbol_billboard.pkl',"
      f" 'model_metadata.json' y 'billboard.csv' disponibles. Error: {e}"
  )
  st.stop()

st.caption(metadata.get("nombre_modelo", "Árbol de Decisión - Billboard Top 1"))

with st.form("datos_cancion"):
  st.subheader("Características de la Nueva Canción")

  col1, col2 = st.columns(2)

  artist_type = col1.selectbox(
      "Tipo de Artista", sorted(df["artist_type"].dropna().unique())
  )
  mood = col2.selectbox("Ánimo (Mood)", sorted(df["mood"].dropna().unique()))

  tempo = col1.selectbox("Tempo", sorted(df["tempo"].dropna().unique()))
  genre = col1.selectbox("Género Musical", sorted(df["genre"].dropna().unique()))

  duration_seg = st.number_input(
      "Duración de la canción (Segundos)",
      min_value=10,
      max_value=1000,
      value=220,
  )

  enviar = st.form_submit_button("Predecir Probabilidad de Éxito")

if enviar:
  nueva_cancion = pd.DataFrame([{
      "artist_type": artist_type,
      "mood": mood,
      "tempo": tempo,
      "genre": genre,
      "durationSeg": float(duration_seg),
  }])

  nueva_cancion_encoded = pd.get_dummies(nueva_cancion, drop_first=True)

  # Extraer automáticamente las columnas exactas que espera el modelo entrenado
  if hasattr(modelo, "feature_names_in_"):
    columnas_modelo_finales = list(modelo.feature_names_in_)
  else:
    columnas_modelo_finales = metadata.get("variables_modelo", [])

  nueva_cancion_encoded = nueva_cancion_encoded.reindex(
      columns=columnas_modelo_finales, fill_value=0
  )

  resultado = int(modelo.predict(nueva_cancion_encoded)[0])

  if resultado == 1:
    st.success(
        "🏆 ¡Pronóstico Altamente Exitoso! Tu canción tiene el potencial para"
        " llegar al TOP 1 de Billboard."
    )
    resultado_texto = "¡Sí, tiene altas probabilidades de llegar al Top 1!"
  else:
    st.warning(
        "🎵 Tu canción tiene un buen ritmo, pero el modelo predice que NO"
        " llegará al TOP 1."
    )
    resultado_texto = "No se pronostica que llegue al Top 1 de Billboard."

  api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

  if api_key:
    prompt = f"""
        Actúa como un productor y analista experto de marketing en la industria discográfica musical.

        Explica brevemente por qué la canción obtuvo este resultado de acuerdo con sus características y brinda 3 recomendaciones prudentes o creativas para maximizar sus posibilidades en las listas.

        Datos de la Canción:
        - Tipo de Artista: {artist_type}
        - Ánimo (Mood): {mood}
        - Tempo: {tempo}
        - Género: {genre}
        - Duración: {duration_seg} segundos

        Resultado del modelo de Árbol de Decisión:
        - Predicción: {resultado_texto}
        """

    try:
      client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

      respuesta = client.chat.completions.create(
          model="llama3-8b-8192",
          messages=[
              {
                  "role": "system",
                  "content": (
                      "Eres un consultor experto de marketing musical de"
                      " Billboard."
                  ),
              },
              {"role": "user", "content": prompt},
          ],
      )

      texto = respuesta.choices[0].message.content
      st.subheader("Análisis del Productor de IA")
      st.info(texto)

    except Exception as e:
      st.warning(f"Error al generar recomendaciones con Groq: {e}")
