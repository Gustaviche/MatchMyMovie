import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://raw.githubusercontent.com/Gustaviche/MatchMyMovie/refs/heads/main/df_v3.csv', index_col=0)
df_actor_pop = pd.read_csv('https://raw.githubusercontent.com/Gustaviche/MatchMyMovie/refs/heads/main/df_actor_actress_updated.csv',index_col=0)
df_director_pop = pd.read_csv('https://raw.githubusercontent.com/Gustaviche/MatchMyMovie/refs/heads/main/director_popularities.csv',index_col=0)


st.set_page_config(
    page_title="MatchMyMovie",
    page_icon="🍿",
    layout="centered",
    initial_sidebar_state="expanded"
)

if 'viewed_movies' not in st.session_state:
    st.session_state['viewed_movies'] = []

def mark_as_seen(movie):
    movie_title = movie['Titre']  # On accède directement à la clé 'Titre'

    # Vérification si la liste viewed_movies existe dans st.session_state
    if 'viewed_movies' not in st.session_state:
        st.session_state.viewed_movies = []

    # Vérification si le film est déjà marqué comme vu
    if movie_title in st.session_state.viewed_movies:
        st.session_state.viewed_movies.remove(movie_title)  # Retirer le titre de la liste des films vus
        st.success(f"Le film **{movie_title}** a été retiré de la liste des films vus.")
    else:
        st.session_state.viewed_movies.append(movie_title)  # Ajouter le titre à la liste des films vus
        st.success(f"Vous avez marqué le film **{movie_title}** comme vu !")


def filter_movies(df, genre=None, acteur=None, real=None, film=None):
    # Initialisation du DataFrame filtré
    filtered_films = df

    # Appliquer le filtre pour le genre si sélectionné
    if genre:
        filtered_films = filtered_films[filtered_films['genres_x'].str.contains(genre, case=False, na=False)].head(10)

    # Appliquer le filtre pour l'acteur si sélectionné
    if acteur:
        filtered_films = filtered_films[filtered_films['cast'].str.contains(acteur, case=False, na=False)].head(10)

    # Appliquer le filtre pour le réalisateur si sélectionné
    if real:
        filtered_films = filtered_films[filtered_films['director'].str.contains(real, case=False, na=False)]

    # Appliquer le filtre pour le titre de film si sélectionné
    if film:
        filtered_films = filtered_films[filtered_films['title_y'].str.contains(film, case=False, na=False)].head(10)

    # Préparer les résultats structurés
    results = []
    if not filtered_films.empty:
        for _, row in filtered_films.iterrows():
            movie_info = {
                'Titre': row['title_y'],
                'Année de sortie': row['startYear'],
                'Genres': row['genres_x'],
                'Réalisateur': row['director'],
                'Acteurs': row['cast'],
                'Synopsis': row['overview'],
                'Affiche': row['poster_path'],
                'Video': row['key']
            }
            results.append(movie_info)

    return results

def filter_and_return_films(df, genre):
    # Initialisation de la liste des résultats
    results = []

    # Filtrer les films en fonction du genre
    if genre:
        filtered_df = df[df['genres_x'].str.contains(genre, case=False, na=False)]
        
        # Trier les films par note moyenne et popularité (en ordre décroissant)
        sorted_df = filtered_df.sort_values(by=['averageRating', 'popularity'], ascending=[False, False])
        
        # Sélectionner les 10 premiers films après tri
        top_10_films = sorted_df.head(10)
        
        # Si des films sont trouvés, les ajouter à la liste des résultats
        for _, movie in top_10_films.iterrows():
                movie_info = {
                    'Titre': movie['title_y'],
                    'Genres': movie['genres_x'],
                    'Année de sortie': movie['startYear'],
                    'Réalisateur': movie['director'],
                    'Acteurs': movie['cast'],
                    'Note moyenne': movie['averageRating'],
                    'Popularité': movie['popularity'],
                    'Affiche': movie['poster_path'],
                    'Synopsis': movie['overview'],
                    'Video' : movie['key']
                }
                results.append(movie_info)

    return results

# Fonction pour recommander des films par mots-clés
def recommend_movies_keyword(df, query, n_neighbors=10, threshold_distance=0.9, stopwords=None):
    if stopwords is None:
        stopwords = ['le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'd\'', 
                     'en', 'que', 'qui', 'à', 'dans', 'sur', 'pour', 'avec', 
                     'sans', 'est', 'et', 'il', 'elle', 'ils', 'elles', 'nous', 
                     'vous', 'ça', 'ce', 'ces']
    
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf_matrix = vectorizer.fit_transform(df['text_concat'])
    
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(tfidf_matrix)
    
    query_tfidf = vectorizer.transform([query])
    distances, indices = knn.kneighbors(query_tfidf)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if dist < threshold_distance:
            movie_info = ({
                'Titre': df.iloc[idx]['title_y'],
                'Année de sortie' : df.iloc[idx]['startYear'],
                'Genres': df.iloc[idx]['genres_x'],
                'Réalisateur': df.iloc[idx]['director'],
                'Acteurs': df.iloc[idx]['cast'],
                'Synopsis': df.iloc[idx]['overview'],
                'Distance': dist,
                'Affiche': df.iloc[idx]['poster_path'],
                'Video': df.iloc[idx]['key']
            })
            results.append(movie_info)

    return results

# Fonction pour recommander des films par genre
def recommend_movies_genres(df, user_genres, n_neighbors=10, metric='cosine'):
    # 1. Préparation des genres
    mlb = MultiLabelBinarizer()

    # Transformer les genres en colonnes binaires
    genre_features = mlb.fit_transform(df['genres_x'].apply(lambda x: x.split(', ')))  # Assurez-vous de séparer les genres en liste
    genre_columns = mlb.classes_  # Noms des colonnes générées

    # Ajouter les colonnes binaires au DataFrame
    df_genres = pd.DataFrame(genre_features, columns=genre_columns)
    df = pd.concat([df, df_genres], axis=1)

    # 2. Création du modèle KNN
    X_genres = df[genre_columns]
    knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    knn_model.fit(X_genres)

    # 3. Créer le vecteur utilisateur basé sur les genres
    user_vector = [1 if genre in user_genres else 0 for genre in genre_columns]
    user_vector_df = pd.DataFrame([user_vector], columns=genre_columns)

    # 4. Trouver les films les plus proches
    distances, indices = knn_model.kneighbors(user_vector_df)

    # 5. Créer des résultats détaillés sous forme de dictionnaires
    results = []

    for i in range(n_neighbors):
        idx = indices[0][i]  # L'index du film recommandé
        row = df.iloc[idx]  # Le film correspondant à cet index
        movie_info = ({
            'Titre': row['title_y'],
            'Année de sortie' : row['startYear'],
            'Genres': row['genres_x'],
            'Réalisateur': row['director'],
            'Acteurs': row['cast'],
            'Synopsis': row['overview'],
            'Affiche': row['poster_path'],
            'Video': row['key']
        })
        results.append(movie_info)

    return results

def background():
    st.markdown("""
    <style>
    /* Fond global pour l'application */
    .stApp {
        background-image: url('https://usbeketrica.com/media/69246/download/3872959826_57043705df_k.jpg?v=1&inline=1');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Ciblage des widgets (conteneur principal) */
    div.block-container {
        background: rgba(255, 255, 255, 0.85); /* Transparence blanche */
        border-radius: 15px; /* Coins arrondis */
        padding: 20px; /* Espacement interne */
    }
    </style>
    """, unsafe_allow_html=True)

def center_content():
    st.markdown("""
    <style>
    .main {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Fonction générique pour afficher les entités (réalisateurs/acteurs) en lignes
def display_entities_in_rows(top_entities, entity_key, photo_column):
    # Définir la répartition des entités par ligne
    entity_rows = [
        [top_entities.iloc[0]],  # Ligne 1 : Top 1
        [top_entities.iloc[1], top_entities.iloc[2]],  # Ligne 2 : Top 2 et 3
        [top_entities.iloc[3], top_entities.iloc[4], top_entities.iloc[5]],  # Ligne 3 : Top 4, 5, 6
        [top_entities.iloc[6], top_entities.iloc[7], top_entities.iloc[8], top_entities.iloc[9]],  # Ligne 4 : Top 7, 8, 9, 10
    ]

    for row in entity_rows:
        cols = st.columns(len(row))  # Créer une colonne par entité dans la ligne
        for i, entity in enumerate(row):
            with cols[i]:
                # Vérifier si l'image existe
                if pd.notna(entity[photo_column]):
                    st.markdown(
                        f"""
                        <div style="text-align: center;">
                            <img src="{entity[photo_column]}" alt="{entity[entity_key]}" width="150">
                            <p><strong>{entity[entity_key]}</strong></p>
                            <p>Nombre de films : {entity['number_of_movies']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="text-align: center;">
                            <p><strong>{entity[entity_key]}</strong></p>
                            <p>Image non disponible</p>
                            <p>Popularité totale : {entity['popularity']}</p>
                            <p>Nombre de films : {entity['number_of_movies']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Fonction générique pour créer un graphique en barres horizontales
def create_bar_chart(data, x_col, y_col, title):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()  # Récupérer l'axe actuel
    
    # Créer un graphique à barres avec Seaborn
    sns.barplot(x=x_col, y=y_col, data=data, ax=ax, palette="viridis")
    
    # Ajouter un titre
    ax.set_title(title)
    
    # Ajouter des étiquettes aux axes
    ax.set_xlabel(y_col)
    ax.set_ylabel(x_col)
    
    # Afficher le graphique
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt) 
# Fonction pour calculer les Top 10 des réalisateurs/acteurs par popularité
def get_top_10_entities_by_popularity(df, entity_key, photo_dict):
    # Grouper par l'entité (réalisateur ou acteur) et compter le nombre de films (lignes) et sommer la popularité
    grouped_entities = df.groupby(entity_key).agg({
        'popularity': 'sum',  # Somme de la popularité pour chaque entité
    }).reset_index()

    # Compter le nombre de films par entité (cela correspond au nombre de lignes par entité)
    grouped_entities['number_of_movies'] = df.groupby(entity_key).size().values
    
    # Trier par popularité décroissante
    top_entities = grouped_entities.sort_values(by='popularity', ascending=False).head(10)
    
    # Ajouter les URLs des photos
    top_entities['photo_url'] = top_entities[entity_key].map(photo_dict)
    
    return top_entities

background()

center_content()

# Menu de navigation
with st.sidebar:
    selection = option_menu(
        menu_title="Menu",  # Titre du menu dans la sidebar
        options=["Accueil",'Presentation Client','Étude de marché','Équipe',"Top 10 des films",'Filtrage des envies', "Films par mots-clés", "Films par genre","Films vus","KPI","Déconnexion"],
        icons=["house-door","person-video","folder2-open","people","bar-chart", "search", "film","camera-reels","clipboard2-check","easel2","power"],
        default_index=0,
        orientation="vertical",  # Menu vertical
        styles={
            "container": {"padding": "5px"},
            "icon": {"font-size": "20px", "color": "#000000"},
            "nav-link": {
                "font-size": "16px", 
                "text-align": "left", 
                "margin": "10px", 
                "--hover-color": "#f08080"
            },
            "nav-link-selected": {"background-color": "#f08080"}
        }
    )

if selection == "Accueil":
    st.title("Bienvenue sur MatchMyMovie")
    st.subheader("Retrouvez des films qui matchent avec vos envies !")
    col1, col2 ,col3, col4 = st.columns(4)
    with col1:
        st.image("https://image.tmdb.org/t/p/original/v0dj9NVPFTX0bv5NtqxK99i1Ae3.jpg") #Mission: Impossible - Rogue Nation 
        st.image("https://image.tmdb.org/t/p/original/sNL1aPGCMFmcnNDFWVUKehO3Vjr.jpg") #OSS 117 : Rio ne répond plus
        st.image("https://image.tmdb.org/t/p/original/dfht1lGq2ALbrRkMj35dUrj5kHG.jpg")  #Bienvenue chez les Ch'tis
    with col2:
        st.image("https://image.tmdb.org/t/p/original/smFyhZHuOCZEmH0kfXOrJLC3Acx.jpg") #Toy Story 4 
        st.image("https://image.tmdb.org/t/p/original/yJm61MmTMjOmNXxPxdoaIkdqnOm.jpg") # Harry Potter and the Deathly Hallows 
        st.image("https://image.tmdb.org/t/p/original/A0Th0x8QIzP0njrFAJnYQ5ouIoB.jpg") #Forrest Gump
    with col3:
        st.image("https://image.tmdb.org/t/p/original/tz4DUBcxU7UowOIJwqvQfdWkU2U.jpg") #The Polar Express
        st.image("https://image.tmdb.org/t/p/original/gdUJ6ECIHNE5M2HImGaBOfb8jR2.jpg")  #Intouchables
        st.image("https://image.tmdb.org/t/p/original/ybjooZMNlRBaFNfs52XqONc4Xyw.jpg")#Deadpool 2"
    with col4:
        st.image("https://image.tmdb.org/t/p/original/obsGPyNOAwkQbLRQOHR6a21VT23.jpg")#Rogue One: A Star Wars Story
        st.image("https://image.tmdb.org/t/p/original/aZ7JWKenzR28H4bCgFJwdCuHovW.jpg")#Babysitting 
        st.image("https://image.tmdb.org/t/p/original/2k0mHrCtIydYR0RA4RyjhRc2hNN.jpg") #Scream'''

elif selection == "Presentation Client":
                st.title("Presentation client")
                st.write("Un cinéma situé dans la Creuse constate une baisse de fréquentation et souhaite moderniser son approche pour attirer et fidéliser une clientèle locale. Le projet consiste à développer un site Internet dédié aux habitants de la région et à mettre en place un moteur de recommandations de films, capable de communiquer avec les clients.")
                st.video("https://www.youtube.com/watch?v=6ocJK8jQ6sI")
    
elif selection == "Étude de marché":
                st.title("Etude de marché")
                st.write("Description :")
                st.write("La Creuse est un département français situé dans la région Nouvelle-Aquitaine et ayant appartenu, avant 2016, à la région Limousin avant la disparition de cette dernière. Il succède à la province de la Marche dont il reprend une grande partie du territoire.")
                st.write("Population : 115 702 hab. (2021)")
                st.write("Capitale : Guéret")
                st.write("Superficie : 5 565 km²")
                st.write("Cantons : 15")
        
                col1, col2 = st.columns(2)
                with col1:
                        st.image("https://www.actualitix.com/wp-content/uploads/2017/05/ou-se-trouve-creuse.jpg")
                        st.write("Localisation de la creuse")
                with col2:
                        st.image("https://upload.wikimedia.org/wikipedia/commons/e/e4/Carte_r%C3%A9sultats_cantons_2015_Creuse.svg")
                        st.write("Cantons de la creuse")
                st.write ("VISUALISATION MARCHE DE LA CREUSE (page 4): https://www.cnc.fr/cinema/etudes-et-rapports/statistiques/datavisualisation-la-geographie-du-cinema")

elif selection == "Équipe":
                st.title("Équipe du projet ")
                col1, col2 = st.columns(2)
                with col1:
                        st.image("https://attic.sh/n7dvxa4xl5bi9pwtg9ntmqhq101b")
                        st.write("Yasmina : Scrum Master")
                        st.image("https://static.tiktokemoji.com/202411/15/x20t8eUr.webp")
                        st.write("Mariétou : Team Member")
                       
                with col2:
                        st.image("https://attic.sh/471qrlom5chx5d65jzw748f3ucf4")
                        st.write("Alexandre : Team Member")
                        st.image("https://attic.sh/78097cw049dbfkjq74trr3gafah0")
                        st.write("Mélanie : Product Owner")#st.image("https://attic.sh/99t8n9w34y6ekmwewzxwk3lx4jbk" )
                
elif selection == "Filtrage des envies":
    st.title("Recommandations")
    st.image("https://img.freepik.com/photos-premium/clap-cinema-loupe-tableau-noir-craie-industrie-du-cinema-divertissement_175682-23332.jpg?w=740")
    
    # Sélection du genre
    genre = st.selectbox("Sélectionner un genre", ["", "Action", 'Adventure', 'Animation', "Comedy", "Crime", "Drama", 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'War', 'Western'])
    
    # Entrée utilisateur pour acteur/actrice et réalisateur/réalisatrice 
    acteur = st.text_input("Saisir un acteur/actrice de votre choix") 
    real = st.text_input("Saisir un réalisateur/réalisatrice de votre choix")
    film = st.text_input("Saisir un nom de film de votre choix")
   
    # Vérification si au moins un critère a été rempli
    if genre or acteur or real or film:
        results = filter_movies(df, genre=genre, acteur=acteur, real=real, film=film)
        if results:
            st.write(f"{len(results)} films trouvés correspondant à vos critères.")
            for movie in results:
                col1, col2 = st.columns([3, 1])  # Ajustez la taille des colonnes
                with col1:
                    st.subheader(movie['Titre'])
                with col2:
                    if st.button(f"J'ai vu", key=f"btn_{movie['Titre']}"):
                        mark_as_seen(movie)
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(movie['Affiche'], caption=movie['Titre'], use_container_width=True)
                with col2:
                    try:
                        # Afficher directement la vidéo YouTube (sans transformation)
                        if movie['Video']:
                            st.video(movie['Video'])
                        else:
                            st.write("Vidéo Indisponible")
                    except Exception as e:
                        st.write("Vidéo Indisponible")
                    
                st.write(f"Genres : {movie['Genres']}")
                st.write(f"Année de sortie : {movie['Année de sortie']}")
                st.write(f"Réalisateur : {movie['Réalisateur']}")
                st.write(f"Acteurs : {movie['Acteurs']}")
                st.write(f"Synopsis : {movie['Synopsis']}")
        else:
            st.warning("Aucun film trouvé correspondant à vos critères.")
    else:
        st.info("Veuillez remplir au moins un des critères pour effectuer une recherche.")

elif selection == "Films par mots-clés":
    st.title("Rechercher un film par mots-clés")
    query = st.text_input("Entrez un ou plusieurs mots-clés")
    
    if query:
        results = recommend_movies_keyword(df, query)
        if results:
            st.write("Films recommandés :")
            for movie in results:
                col1, col2 = st.columns([3, 1])  # Ajustez la taille des colonnes
                with col1:
                    st.subheader(movie['Titre'])
                with col2:
                    if st.button(f"J'ai vu", key=f"btn_{movie['Titre']}"):
                        mark_as_seen(movie)
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(movie['Affiche'], caption=movie['Titre'], use_container_width=True)
                with col2:
                    try:
                        # Afficher directement la vidéo YouTube (sans transformation)
                        if movie['Video']:
                            st.video(movie['Video'])
                        else:
                            st.write("Vidéo Indisponible")
                    except Exception as e:
                        st.write("Vidéo Indisponible")
                    
                st.write(f"Genres : {movie['Genres'].replace("[", "").replace("]", "").replace("'", "").replace('"',"")}")
                st.write(f"Année de sortie : {movie['Année de sortie']}")
                st.write(f"Réalisateur : {movie['Réalisateur'].replace("[", "").replace("]", "").replace("'", "").replace('"',"")}")
                st.write(f"Acteurs : {movie['Acteurs'].replace("[", "").replace("]", "").replace("'", "").replace('"',"")}")
                st.write(f"Synopsis : {movie['Synopsis']}")
                st.markdown("<hr style='border: 2px solid #f08080;'>", unsafe_allow_html=True)
        else:
            st.write("Aucun film trouvé avec ces mots-clés.")
    
elif selection == "Films par genre":
    st.title("Recommandation de films par genre")
    
    # Sélection du premier genre
    genre_1 = st.selectbox("Sélectionner un genre", ["", "Action", 'Adventure', 'Animation', "Comedy", "Crime", "Drama", 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'War', 'Western'])
    
    # Sélection du deuxième genre (optionnel)
    genre_2 = st.selectbox("Sélectionner un deuxième genre (optionnel)", ["", "Action", 'Adventure', 'Animation', "Comedy", "Crime", "Drama", 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'War', 'Western'])
    
    # Création de la liste des genres sélectionnés
    user_genres = [genre_1]
    if genre_2:
        user_genres.append(genre_2)

    # Vérification si au moins un genre a été sélectionné
    if user_genres and "" not in user_genres:
        # Appeler la fonction de recommandation avec les genres sélectionnés
        recommended_movies = recommend_movies_genres(df, user_genres)
        
        if recommended_movies:
            st.write("Films recommandés par genre :")
            for movie in recommended_movies:
                col1, col2 = st.columns([3, 1])  # Ajuster la taille des colonnes
                with col1:
                    st.subheader(movie['Titre'])
                with col2:
                    if st.button(f"J'ai vu", key=f"btn_{movie['Titre']}"):
                        mark_as_seen(movie)
                
                # Affichage de l'affiche et de la vidéo
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(movie['Affiche'], caption=movie['Titre'], use_container_width=True)
                with col2:
                    try:
                        if movie['Video']:
                            st.video(movie['Video'])
                        else:
                            st.write("Vidéo Indisponible")
                    except Exception as e:
                        st.write("Vidéo Indisponible")
                
                # Affichage des détails du film
                genres = ", ".join(movie['Genres']) if isinstance(movie['Genres'], list) else movie['Genres']
                st.write(f"Genres : {genres}")
                st.write(f"Année de sortie : {movie['Année de sortie']}")
                st.write(f"Réalisateur : {movie['Réalisateur'].replace('[', '').replace(']', '').replace("'", "").replace('\"', '')}")
                st.write(f"Acteurs : {movie['Acteurs'].replace('[', '').replace(']', '').replace('\'', '').replace('\"', '')}")
                st.write(f"Synopsis : {movie['Synopsis']}")
                
                # Ajouter une ligne horizontale pour séparer les films
                st.markdown("<hr style='border: 2px solid #f08080;'>", unsafe_allow_html=True)
        else:
            st.write("Aucun film trouvé pour ces genres.")
    else:
        st.write("Veuillez sélectionner au moins un genre pour voir les recommandations.")

elif selection == "Top 10 des films":
    st.title("Top 10 des meilleurs films")
    st.image("https://th.bing.com/th/id/R.7e9183f95a1237eed74b6e911f4aa030?rik=6FRjaPSeXY2pUw&riu=http%3a%2f%2fmariepierrem.m.a.pic.centerblog.net%2ffv5s7fd8s6.png&ehk=LJjgr%2bndWf9891%2bDJgWc5f5Gm9ckXgW4Q6RAZC%2b0c%2fA%3d&risl=&pid=ImgRaw&r=0&sres=1&sresct=1", use_container_width=True)
    st.subheader("Choisissez un genre pour découvrir les meilleurs films du genre")

    # Liste des genres
    genres = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "Thriller", "War", "Western"]

    # Sélection du genre avec st.selectbox (ou st.radio si vous préférez)
    genre = st.segmented_control("Choisissez un genre", options=genres)

    # Appeler la fonction pour filtrer et obtenir les films
    top_10_films = filter_and_return_films(df, genre)

    # Vérifier si des films sont trouvés et les afficher
    if top_10_films:
        st.write(f"### Top 10 des films pour le genre : **{genre}**")
        for movie in top_10_films:
            # Affichage dans deux colonnes pour l'image et le bouton
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"**{movie['Titre']}**")
            
            with col2:
                # Affichage du bouton "J'ai vu"
                if st.button(f"J'ai vu", key=f"btn_{movie['Titre']}"):
                    mark_as_seen(movie)  # Passer le dictionnaire complet du film
                    
            # Affichage des informations détaillées
            col1, col2 = st.columns([1, 2])
            with col1:
                # Affichage de l'image du film
                st.image(movie['Affiche'], use_container_width=True)
            with col2:
                try:
                    if movie['Video']:
                        st.video(movie['Video'])
                    else:
                        st.write("Vidéo Indisponible")
                except Exception as e:
                    st.write("Vidéo Indisponible")
                
            # Informations sur le film
            st.write(f"**Genres :** {movie['Genres']}")
            st.write(f"**Réalisateur :** {movie['Réalisateur']}")
            st.write(f"**Acteurs :** {movie['Acteurs']}")
            st.write(f"**Note moyenne :** {movie['Note moyenne']}")
            st.write(f"**Popularité :** {movie['Popularité']}")
            st.write(f"**Synopsis :** {movie['Synopsis']}")
            
            st.markdown("<hr style='border: 2px solid #f08080;'>", unsafe_allow_html=True)
    else:
        st.write(f"Sélectionnez un genre")

elif selection == "Films vus":
    # Affichage des films vus
    st.title("Films déjà vus")
    
    # Si la liste des films vus contient des films
    if st.session_state.viewed_movies:
        st.write("Voici les films que vous avez vus :")
        
        # Parcours de chaque titre de film dans la liste des films vus
        for movie_title in st.session_state.viewed_movies:
            # Recherche dans le DataFrame pour obtenir les détails du film
            movie = df[df['title_y'] == movie_title].iloc[0]  # Assurez-vous qu'il y a au moins un film trouvé
            
            # Utilisation de la mise en page avec 2 colonnes pour plus de clarté
            col1, col2 = st.columns([1, 2])  # Ajuster la proportion des colonnes
            with col1:
                # Affichage du titre et de l'image
                if movie['poster_path']:  # Vérification si l'URL de l'affiche existe
                    st.image(movie['poster_path'], caption=movie['title_y'], use_container_width=True)
                
            with col2:
                # Affichage des informations supplémentaires du film
                st.write(f"**Genres :** {movie['genres_x']}")
                st.write(f"**Réalisateur :** {movie['director']}")
                st.write(f"**Acteurs :** {movie['cast']}")
                st.write(f"**Synopsis :** {movie['overview']}")
            
            st.markdown("<hr style='border: 2px solid #f08080;'>", unsafe_allow_html=True)  # Séparation visuelle entre les films
    else:
        st.write("Vous n'avez pas encore marqué de films comme vus.")

elif selection == "KPI":
    director_photos = {
                    "Steven Spielberg": "https://image.tmdb.org/t/p/w500/tZxcg19YQ3e8fJ0pOs7hjlnmmr6.jpg",
                    "Clint Eastwood": "https://image.tmdb.org/t/p/w500/dU35NnjZ4aGw5abIJe3WXVf3Eey.jpg",
                    "Quentin Tarantino": "https://image.tmdb.org/t/p/w500/1gjcpAa99FAOWGnrUvHEXXsRs7o.jpg",
                    "Ridley Scott": "https://image.tmdb.org/t/p/w500/zABJmN9opmqD4orWl3KSdCaSo7Q.jpg",
                    "Martin Scorsese": "https://image.tmdb.org/t/p/w500/mN49M9y74NZgMVKN35Qan5cfxrP.jpg",
                    "Christopher Nolan": "https://image.tmdb.org/t/p/w500/xuAIuYSmsUzKlUMBFGVZaWsY3DZ.jpg",
                    "Tim Burton": "https://image.tmdb.org/t/p/w500/wcjuY5vD1nlfwWNbvvTGg5dGoRR.jpg",
                    "Sylvester Stallone": "https://image.tmdb.org/t/p/w500/gn3pDWthJqR0VDYGViGD3048og7.jpg",
                    "Ron Howard": "https://image.tmdb.org/t/p/w500/tDKn6gAgvARdQRfuem2JwyNcP3B.jpg",
                    "Robert Zemeckis": "https://image.tmdb.org/t/p/w500/lPYDQ5LYNJ12rJZENtyASmVZ1Ql.jpg",
                     }

    actor_photos = {
                    "Tom Hanks": "https://image.tmdb.org/t/p/w500/eKF1sGJRrZJbfBG1KirPt1cfNd3.jpg",
                    "Robert De Niro": "https://image.tmdb.org/t/p/w500/cT8htcckIuyI1Lqwt1CvD02ynTh.jpg",
                    "Chris Evans": "https://image.tmdb.org/t/p/w500/3bOGNsHlrswhyW79uvIHH1V43JI.jpg",
                    "Tom Cruise": "https://image.tmdb.org/t/p/w500/8qBylBsQf4llkGrWR3qAsOtOU8O.jpg",
                    "Hugh Jackman": "https://image.tmdb.org/t/p/w500/4Xujtewxqt6aU0Y81tsS9gkjizk.jpg",
                    "Robert Downey Jr.": "https://image.tmdb.org/t/p/w500/5qHNjhtjMD4YWH3UP0rm4tKwxCL.jpg",
                    "Matt Damon": "https://image.tmdb.org/t/p/w500/vgqgSZvrIrhY5yngVIg1R1KZQWQ.jpg",
                    "Samuel L. Jackson": "https://image.tmdb.org/t/p/w500/AiAYAqwpM5xmiFrAIeQvUXDCVvo.jpg",
                    "Mark Ruffalo": "https://image.tmdb.org/t/p/w500/5GilHMOt5PAQh6rlUKZzGmaKEI7.jpg",
                    "Chris Hemsworth": "https://image.tmdb.org/t/p/w500/u9Ovh0g3b9kqYjcydqW11PCv5DZ.jpg"
                     }
    
    # Streamlit - Interface
    st.title("Visualisation des KPI")

    option = st.selectbox("Sélectionnez la Visualisation", ("Top 10 Réalisateurs par Popularité", "Top 10 Acteurs/Actrices par Popularité","Nombre de Films par Année","Films les plus populaires par Année"))

    # Traitement de l'option sélectionnée
    if option == "Top 10 Réalisateurs par Popularité":
        top_directors = get_top_10_entities_by_popularity(df_director_pop, 'director', director_photos)
        st.subheader("Top 10 Réalisateurs par Popularité")
        display_entities_in_rows(top_directors, 'director', 'photo_url')
        st.subheader("Graphique des Top 10 Réalisateurs par Popularité")
        create_bar_chart(top_directors, 'director', 'popularity', "Top 10 Réalisateurs par Popularité")

    elif option == "Top 10 Acteurs/Actrices par Popularité":
        top_actors = get_top_10_entities_by_popularity(df_actor_pop, 'actor_actress', actor_photos)
        st.subheader("Top 10 Acteurs/Actrices par Popularité")
        display_entities_in_rows(top_actors, 'actor_actress', 'photo_url')
        st.subheader("Graphique des Top 10 Acteurs/Actrices par Popularité")
        create_bar_chart(top_actors, 'actor_actress', 'popularity', "Top 10 Acteurs par Popularité")
    
    elif option == 'Nombre de Films par Année':

                        # Sélection des critères par l'utilisateur
                        threshold = st.slider("Seuil de Note Moyenne (averageRating)", 0.0, 10.0, 7.5)
                        year_range = st.slider("Années", min_value=int(df['startYear'].min()), max_value=int(df['startYear'].max()), value=(2010, 2024))

                        # Filtrer les films selon le seuil et la plage d'années
                        filtered_films = df[(df['averageRating'] > threshold) & (df['startYear'].between(year_range[0], year_range[1]))]

                        # Compter le nombre de films par année
                        films_per_year = filtered_films.groupby('startYear').size()

                        # Afficher les résultats
                        #st.write(f"Nombre de films par année avec une note supérieure à {threshold} entre {year_range[0]} et {year_range[1]}")
                        #st.write(films_per_year)

                        # Visualiser les résultats
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.barplot(x=films_per_year.index, y=films_per_year.values, palette='viridis', ax=ax)
                        ax.set_title(f'Nombre de Films par Année à {threshold}')
                        ax.set_xlabel('Année')
                        ax.set_ylabel('Nombre de Films')
                        plt.xticks(rotation=45)

                        # Afficher le graphique dans Streamlit
                        st.pyplot(fig)

                        # Afficher la liste des films correspondant aux critères 
                        st.write(f"Liste des films avec une note supérieure à {threshold} entre {year_range[0]} et {year_range[1]}:") 

                        # Itérer sur les années et afficher les films correspondants 
                        for year in range(year_range[0], year_range[1] + 1): 
                            st.write(f"Année {year}:") 
                            films_of_year = filtered_films[filtered_films['startYear'] == year] 
                            for index, row in films_of_year.iterrows(): 
                                  st.write(f"- {row['title_y']} (Note: {row['averageRating']})")

    elif option == "Films les plus populaires par Année": 
                    
                    st.title("Top 10 Films les Plus Populaires")

                    # Laisser l'utilisateur choisir l'année
                    year_choice = st.slider("Choisissez une année", min_value=1960, max_value=2024, step=1)

                    # Filtrer les films de l'année choisie
                    df_filtered = df[df['startYear'] == year_choice]

                    # Trier les films par nombre de votes et sélectionner les 10 meilleurs
                    df_top10 = df_filtered.sort_values(by='averageRating', ascending=False).head(10)

                    # Afficher les posters des 10 films les plus populaires en 5 colonnes
                    st.write(f"Top 10 des films avec le plus de votes en {year_choice}")
                    cols = st.columns(5)

                    for i, row in enumerate(df_top10.iterrows()): 
                           with cols[i % 5]: st.image(row[1]['poster_path'], caption=row[1]['title_y']) 
                           # Séparer les lignes 
                           if i == 4: 
                                 cols = st.columns(5)

elif selection == "Déconnexion":
    st.write("Vous êtes déconnecté.")
