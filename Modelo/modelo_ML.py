# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import MultiLabelBinarizer


# movies_filt = pd.read_parquet('../datasets/movie_modelo.parquet')
# # corregir el modelo
# #PREPROCESAMIENTO
# # Convertir columna 'genres' de una cadena de texto a una lista
# movies_filt['genres'] = movies_filt['genres'].apply(lambda x: x.split(','))  # Asegúrate de que los géneros estén separados por comas


# # Se codifica los géneros a binario
# mlb = MultiLabelBinarizer()
# genres_encoded = mlb.fit_transform(movies_filt['genres'])
# genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)


# # Seleccionar los features numericos
# features_num = movies_filt[['vote_average', 'popularity']]

# # Se normaliza escalando
# scaler = StandardScaler()
# features_num_escaldas = scaler.fit_transform(features_num)
# numeric_df = pd.DataFrame(features_num_escaldas, columns=features_num.columns)

# # Se concatena los features normalizados y escalados
# features_df = pd.concat([genres_df.reset_index(drop=True), numeric_df.reset_index(drop=True)], axis=1)

# # calculo de la matriz
# cosine_simi = cosine_similarity(features_df, features_df)

# # FUNCION DE RECOMENDACION

# def recomendacion(titulo, df):

#     # Normalizar el título para la búsqueda
#     titulo = titulo.title()

#     titulo_row = df[df["title"] == titulo ]
    
#     # Se verifica si la película existe en la base de datos
#     if titulo_row.empty:
#         return 0

#     # Se obtiene el indice dado el titulo
#     idx = titulo_row.index[0]

#     # Obtengo los puntajes de similitud 
#     sim_scores = list(enumerate(cosine_simi[idx]))

#     # Se ordena según la similitud de puntajes en orden descendente
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

#     # Indices de las películas similares
#     sim_indices = [i[0] for i in sim_scores if i[0] != idx]

#     # Se verifica si peliculas similares
#     if len(sim_indices) == 0:
#         return f"No se encontraron películas similares a '{titulo}'."

#     # Obtengo los titulos de las 5 películas más similares
#     top_movies = df['title'].iloc[sim_indices[:5]].values.tolist()

#     return top_movies

# # titulo_pelicula = 'Toy Story'
# # recomendaciones = recomendacion(titulo_pelicula, movies_filt)


from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd

movies_filt = pd.read_parquet('../datasets/movie_modelo.parquet')
# # Normalización de popularity, budget, revenue y vote_average
# df= movies_filt
# # Supón que tienes el DataFrame `df` con las columnas mencionadas
# scaler = MinMaxScaler()
# df[['popularity', 'budget', 'revenue', 'vote_average']] = scaler.fit_transform(
#     df[['popularity', 'budget', 'revenue', 'vote_average']]
# )


# # Procesamiento de overview (sinopsis)

# # Limita la cantidad de palabras para reducir dimensiones
# tfidf = TfidfVectorizer(max_features=100)
# tfidf_matrix = tfidf.fit_transform(df['overview']).toarray()


# # Concatenación de todas las características
# # Combina las características numéricas y los vectores TF-IDF de la sinopsis
# X = np.concatenate([
#     df[['popularity', 'budget', 'revenue', 'vote_average']].values,
#     tfidf_matrix
# ], axis=1)


# # SIMILITUD DEL COSENO

# # Calcula la matriz de similitud de coseno
# cosine_sim = cosine_similarity(X)

# # Función para obtener las películas más similares
# def recomendacion(movie, cosine_sim_matrix, df, top_n=5):
#     """
#     Devuelve una lista numerada con las películas más similares a la película especificada.
    
#     Inputs:
#     - movie_index (int): Índice de la película base en el DataFrame.
#     - cosine_sim_matrix (array): Matriz de similitud de coseno.
#     - df (DataFrame): DataFrame que contiene las películas.
#     - top_n (int): Número de recomendaciones (default=5).
    
#     Outputs:
#     - list: Lista de recomendaciones en formato numerado.
#     """
#    # Verifica si el título de la película existe en el DataFrame
#     if movie not in df['title'].values:
#         return [f"La película '{movie}' no se encontró en la base de datos."]
    
#     # Obtiene el índice de la película con el título dado
#     movie_index = df[df['title'] == movie].index[0]


#     # Calcula la similitud de la película base con todas las demás
#     sim_scores = list(enumerate(cosine_sim_matrix[movie_index]))
    
#     # Ordena las películas por similitud (de mayor a menor) y omite la película base
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_indices = [i[0] for i in sim_scores[1:top_n+1]]
    
#     # Extrae los títulos de las películas recomendadas
#     titulos_recom = df.iloc[sim_indices]['title'].tolist()
    
#     # Formatea la lista en formato numerado
#     list_enumerada = [f"{i+1}. {title}" for i, title in enumerate(titulos_recom)]
    
#     return list_enumerada
# # Ejemplo: muestra las 5 películas más similares a la del índice 0
# funcion_ml = recomendacion('Toy Story', cosine_sim, df)

# for movie in funcion_ml:
#     print(movie)