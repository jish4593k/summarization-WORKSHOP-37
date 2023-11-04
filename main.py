import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import numpy as np
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow import keras

# Download NLTK data (one-time execution)
nltk.download('punkt')
nltk.download('stopwords')

# Load your text data
def load_text_data():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            return text
    return None

# Preprocess the text data
def preprocess_text(text):
    sentences = sent_tokenize(text)
    corpus = []
    for sentence in sentences:
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        sentence = sentence.lower()
        sentence = sentence.split()
        sentence = ' '.join([word for word in sentence if word not in stopwords.words('english')])
        corpus.append(sentence)
    return corpus

# Create word embeddings using Keras
def create_word_embeddings(corpus, num_words=1000, max_sequence_length=100):
    tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
    tokenizer.fit_on_texts(corpus)
    sequences = tokenizer.texts_to_sequences(corpus)
    word_index = tokenizer.word_index
    X = pad_sequences(sequences, maxlen=max_sequence_length)
    return X, word_index

# Perform K-means clustering
def kmeans_clustering(X, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    labels = kmeans.labels_
    return labels

# Visualize clustering results
def visualize_clusters(X, labels):
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2)

    pca_result = pca.fit_transform(X)
    tsne_result = tsne.fit_transform(X)

    df = pd.DataFrame({'x': tsne_result[:,0], 'y': tsne_result[:,1], 'label': labels})
    sns.lmplot(x='x', y='y', data=df, fit_reg=False, hue='label', legend=True)
    plt.show()

# Show the most representative sentence in each cluster
def show_representative_sentences(text, labels, num_clusters):
    representative_sentences = []
    cluster_centers = []

    for cluster in range(num_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        cluster_centers.append(np.mean(cluster_indices))

    cluster_centers = [int(round(center)) for center in cluster_centers]
    for center in cluster_centers:
        representative_sentences.append(text[center])

    return representative_sentences

# Create a Tkinter GUI
root = tk.Tk()
root.title("Sentence Clustering")

# Load Text Data
text_data = None

def load_text_button():
    global text_data
    text_data = load_text_data()
    if text_data:
        messagebox.showinfo("Success", "Text data loaded successfully!")

load_text_button = tk.Button(root, text="Load Text Data", command=load_text_button)
load_text_button.pack(pady=10)

# Sentence Clustering
sentence_clusters = None

def cluster_sentences():
    global text_data, sentence_clusters

    if text_data is None:
        messagebox.showerror("Error", "Text data is not loaded!")
        return

    num_clusters = num_clusters_entry.get()
    if not num_clusters.isdigit() or int(num_clusters) < 1:
        messagebox.showerror("Error", "Invalid number of clusters!")
        return

    text_corpus = preprocess_text(text_data)
    X, word_index = create_word_embeddings(text_corpus)
    labels = kmeans_clustering(X, int(num_clusters))
    sentence_clusters = show_representative_sentences(text_corpus, labels, int(num_clusters))
    visualize_clusters(X, labels)

cluster_button = tk.Button(root, text="Cluster Sentences", command=cluster_sentences)
cluster_button.pack(pady=10)

# Number of Clusters Entry
num_clusters_label = tk.Label(root, text="Number of Clusters:")
num_clusters_label.pack()
num_clusters_entry = tk.Entry(root)
num_clusters_entry.pack()

# Show Representative Sentences
def show_representatives():
    global sentence_clusters
    if sentence_clusters:
        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        for i, sentence in enumerate(sentence_clusters):
            result_text.insert(tk.END, f"Cluster {i + 1}: {sentence}\n\n")
        result_text.config(state=tk.DISABLED)

show_representatives_button = tk.Button(root, text="Show Representative Sentences", command=show_representatives)
show_representatives_button.pack(pady=10)

# Result Text Box
result_text = scrolledtext.ScrolledText(root, height=10, width=40, wrap=tk.WORD, state=tk.DISABLED)
result_text.pack()

root.mainloop()
