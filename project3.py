import sys
import os
import argparse
import pickle
import nltk
import pandas as pd
import numpy as np
import en_core_web_lg
from pypdf import PdfReader
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.sparse.linalg import svds
from text_normalizer import normalize_corpus

def extract_text(file_path):
    pdf_file = open(file_path, 'rb')
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() 
    pdf_file.close()
    return text

def extract_state_city_name(file):
    if "DC" in file:
        return "DC",  "Washington, D.C"
    parts = file.split(' ')
    pdf_extension_remove = parts[-1].split('.')
    pdf_extension_remove.pop(-1)
    state = parts[0]
    del parts[0]
    parts.pop(-1)
    parts = parts + pdf_extension_remove
    sep = ' '
    city_name = sep.join(parts)
    return state, city_name

def get_most_common_words(text, n=10):
    word_frequencies = Counter(text.split())
    most_common_words = word_frequencies.most_common(n)
    return [word for word, count in most_common_words]

def remove_most_common_words(text, most_common_words):
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token not in most_common_words]
    return ' '.join(filtered_tokens)

def low_rank_svd(matrix, singular_count=2):
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt

def get_top_words_for_topic(vt, feature_names, num_top_words=5):
    top_words_indices = (-vt).argsort()[:, :num_top_words]
    top_words = [[feature_names[index] for index in topic] for topic in top_words_indices]
    return top_words

def remove_city_state_names(top_words, city_names, state_names):
    city_state_names = [name.lower() for name in city_names + state_names]
    filtered_top_words = [word for word in top_words if word not in city_state_names]
    return filtered_top_words

def correct_words(words, nlp):
    corrected_words = []
    for word in words:
        doc = nlp(word)
        if len(doc) > 0 and doc[0].has_vector:
            token = doc[0].text
            corrected_words.append(token)
    return corrected_words

def main(arg_file_name):
    args = parser.parse_args()
    files_path = os.path.join(os.getcwd(), 'smartcity')
    pdf_files = [file for file in os.listdir(files_path) if file.endswith('.pdf')]
    if args.document not in pdf_files:
        print(f"The document '{args.document}' was not found in the 'smartcity' subdirectory.")
        sys.exit(1)
    
    arg_state, arg_city = extract_state_city_name(arg_file_name)
    data = []
    for pdf in pdf_files:
        state, city = extract_state_city_name(pdf)
        try:
            raw_text = extract_text(os.path.join(files_path, pdf))
            data.append([state, city, raw_text])
        except:
            print(pdf)
    df = pd.DataFrame(data, columns = ['State','City', 'Raw Text'])

    custom_stopwords = ["smart", "city", "page", "content", "appendix", ""]
    city_names = df['City'].apply(lambda x: x.lower().split()).tolist()
    state_abbv = df['State'].apply(lambda x: x.lower().split()).tolist()
    state_names = [
        'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware', 'florida',
        'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 'maine',
        'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi', 'missouri', 'montana', 'nebraska',
        'nevada', 'new hampshire', 'new jersey', 'new mexico', 'new york', 'north carolina', 'north dakota', 'ohio',
        'oklahoma', 'oregon', 'pennsylvania', 'rhode island', 'south carolina', 'south dakota', 'tennessee', 'texas',
        'utah', 'vermont', 'virginia', 'washington', 'west virginia', 'wisconsin', 'wyoming'
    ]
    custom_stopwords.extend(city_names + state_names + state_abbv)
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(custom_stopwords)
    df['Initial Cleaned Text'] = normalize_corpus(df['Raw Text'], stopwords=stopwords)
    df['Most Common Words'] = df['Initial Cleaned Text'].apply(get_most_common_words) 
    df['Final Cleaned Text'] = df.apply(lambda row: remove_most_common_words(row['Initial Cleaned Text'], row['Most Common Words']), axis=1)


    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Final Cleaned Text'])
    with open('model.pkl', 'rb') as file:
        hierarchical_model = pickle.load(file)
    hierarchical_labels = hierarchical_model.fit_predict(X.toarray())
    df['Cluster ID'] = hierarchical_labels

    num_topics = 36
    u, s, vt = low_rank_svd(X.toarray(), singular_count=num_topics)

    feature_names = vectorizer.get_feature_names_out()
    top_words_per_topic = get_top_words_for_topic(vt, feature_names, num_top_words=5)
    nlp = en_core_web_lg.load()
    city_names = df['City'].unique().tolist()
    state_names = df['State'].unique().tolist()
    filtered_and_corrected_top_words = []

    for i, top_words in enumerate(top_words_per_topic):
        filtered_words = remove_city_state_names(top_words, city_names, state_names)
        corrected_words = correct_words(filtered_words, nlp)
        filtered_and_corrected_top_words.append(corrected_words)

    cluster_topic_scores = np.zeros((len(np.unique(hierarchical_labels)), num_topics))

    for cluster_id in range(len(np.unique(hierarchical_labels))):
        cluster_cities_indices = np.where(hierarchical_labels == cluster_id)
        cluster_topic_scores[cluster_id] = u[cluster_cities_indices].mean(axis=0)

    top_two_topics_df = pd.DataFrame(columns=['Top Topic 1', 'Top Topic 2'])


    for city_index in range(df.shape[0]):
        cluster_id = df.loc[city_index, 'Cluster ID']
        cluster_topics = cluster_topic_scores[cluster_id]
        sorted_topics_indices = np.argsort(cluster_topics)[::-1][:2]
        top_two_topics = [filtered_and_corrected_top_words[topic] for topic in sorted_topics_indices]
        top_two_topics_df.loc[city_index] = [', '.join(top_two_topics[0]), ', '.join(top_two_topics[1])]

    df['Top Topic 1'] = top_two_topics_df['Top Topic 1']
    df['Top Topic 2'] = top_two_topics_df['Top Topic 2']

    city_row = df[df['City'] == arg_city].iloc[0]
    print(f"{arg_city}, {arg_state}, cluster_id: {city_row['Cluster ID']}")
    output_file = "smartcity_predict.tsv"
    if not os.path.isfile(output_file):
        with open(output_file, "w") as file:
            file.write("City\tRaw Text\tClean Text\tCluster ID\n")
    
    with open(output_file, "a") as file:
        file.write(f"{arg_city}, {arg_state}\t{city_row['Raw Text']}\t{city_row['Final Cleaned Text']}\t{city_row['Cluster ID']}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", type=str, required=True, 
                         help="full document name include extension")
    args = parser.parse_args()

    if args.document:
        main(args.document)
    else: 
        print("Error: --document argument is required.")
        print("Please rerun the application with the correct argument.")
        exit()