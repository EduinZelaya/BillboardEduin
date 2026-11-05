app_code = """import os
import json
import joblib
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title='Predicción Billboard', layout='centered')
st.title('Predicción de Éxito Billboard Top 1 - Tu Nombre - ISC')

@st.cache_resource
def cargar_modelo():
    pkl = '/content/models/arbol_billboard.pkl' if os.path.exists('/content/models/arbol_billboard.pkl') else 'arbol_billboard.pkl'
    meta = '/content/models/model_metadata.json' if os.path.exists('/content/models/model_metadata.json') else 'model_metadata.json'
    
    modelo = joblib.load(pkl)

    with open(meta, encoding='utf-8') as f:
        metadata = json.load(f)

    return modelo, metadata

@st.cache_data
def cargar_base():
    csv = '/content/billboard.csv' if os.path.exists('/content/billboard.csv') else 'billboard.csv'
    return pd.read_csv(csv)

# Intentar cargar recursos
try:
    modelo, metadata = cargar_modelo()
    df = cargar_base()
    # Columnas de entrenamiento originales utilizadas para el reindex con One-Hot Encoding
    columnas_entrenamiento = metadata.get('variables_modelo', ['artist_type', 'mood', 'tempo', 'genre', 'durationSeg'])
except Exception as e:
    st.error(f"Error al cargar los archivos. Asegúrate de tener 'arbol_billboard.pkl', 'model_metadata.json' y 'billboard.csv' disponibles. Error: {e}")
    st.stop()

st.caption(metadata.get('nombre_modelo', 'Árbol de Decisión - Billboard Top 1'))

with st.form('datos_cancion'):
    st.subheader("Características de la Nueva Canción")
    
    col1, col2 = st.columns(2)

    # Entradas basadas en los valores únicos del dataset original
    artist_type = col1.selectbox('Tipo de Artista', sorted(df['artist_type'].dropna().unique()))
    mood = col2.selectbox('Ánimo (Mood)', sorted(df['mood'].dropna().unique()))

    tempo = col1.selectbox('Tempo', sorted(df['tempo'].dropna().unique()))
    genre = col2.selectbox('Género Musical', sorted(df['genre'].dropna().unique()))

    duration_seg = st.number_input('Duración de la canción (Segundos)', min_value=10, max_value=1000, value=220)

    enviar = st.form_submit_button('Predecir Probabilidad de Éxito')

if enviar:
    # 1. Crear el DataFrame inicial de una fila con las respuestas seleccionadas
    nueva_cancion = pd.DataFrame([{
        'artist_type': artist_type,
        'mood': mood,
        'tempo': tempo,
        'genre': genre,
        'durationSeg': float(duration_seg)
    }])

    # 2. Aplicar One-Hot Encoding (tal como se hizo al entrenar)
    nueva_cancion_encoded = pd.get_dummies(nueva_cancion, drop_first=True)
    
    # 3. Reindexar para asegurar que tiene exactamente las mismas columnas (e.g. 35 variables finales)
    # Nota: En tu entrenamiento original, las variables de entrenamiento son las columnas de 'X'
    # Para evitar incongruencias, usamos las columnas del modelo entrenado definidas en el Notebook.
    # Si no estuvieran en la metadata de forma directa, recreamos la estructura esperada de variables.
    # Vamos a usar las columnas del DataFrame de entrenamiento 'X' que creamos en el notebook actual.
    columnas_modelo_finales = [
        'durationSeg', 'artist_type_Male', 'artist_type_Mixed', 
        'mood_Brooding', 'mood_Cool', 'mood_Defiant', 'mood_Easygoing', 
        'mood_Empowering', 'mood_Energizing', 'mood_Excited', 'mood_Fiery', 
        'mood_Grizzled', 'mood_Hypnotic', 'mood_Lively', 'mood_Melancholy', 
        'mood_Mysterious', 'mood_Optimistic', 'mood_Other', 'mood_Peaceful', 
        'mood_Rowdy', 'mood_Sensual', 'mood_Sophisticated', 'mood_Spirited', 
        'mood_Stirring', 'mood_Urgent', 'mood_Yearning', 'tempo_Medium Tempo', 
        'tempo_Slow Tempo', 'genre_Electronica', 'genre_Jazz', 'genre_Other', 
        'genre_Pop', 'genre_Rock', 'genre_Soundtrack', 'genre_Traditional', 'genre_Urban'
    ]
    
    nueva_cancion_encoded = nueva_cancion_encoded.reindex(columns=columnas_modelo_finales, fill_value=0)

    # Realizar predicción
    resultado = int(modelo.predict(nueva_cancion_encoded)[0])

    if resultado == 1:
        st.success('🏆 ¡Pronóstico Altamente Exitoso! Tu canción tiene el potencial para llegar al TOP 1 de Billboard.')
        resultado_texto = "¡Sí, tiene altas probabilidades de llegar al Top 1!"
    else:
        st.warning('🎵 Tu canción tiene un buen ritmo, pero el modelo predice que NO llegará al TOP 1.')
        resultado_texto = "No se pronostica que llegue al Top 1 de Billboard."

    # Integración de asesoría de IA (Groq)
    api_key = st.secrets.get('GROQ_API_KEY', os.getenv('GROQ_API_KEY', ''))

    if api_key:
        prompt = f'''
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
        '''

        try:
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )

            respuesta = client.chat.completions.create(
                model="llama3-8b-8192", # Actualizado a un modelo rápido y estable de Groq
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un consultor experto de marketing musical de Billboard."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            texto = respuesta.choices[0].message.content
            st.subheader("Análisis del Productor de IA")
            st.info(texto)

        except Exception as e:
            st.warning(f'Error al generar recomendaciones con Groq: {e}')
    else:
        st.warning('Agrega la variable GROQ_API_KEY en los secretos de Streamlit para recibir retroalimentación del Productor de IA.')

st.divider()
st.write('Vista rápida de canciones históricas (Billboard):')
st.dataframe(df.head(20), use_container_width=True)
"""

# Guardar el archivo app.py en el entorno local
with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

print("Archivo 'app.py' creado y guardado con éxito.")