import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA

columns = ["id", "phrase", "profession", "author"]
data = pd.read_csv("./data.txt", names=columns)

target = "author"
x1 = "phrase"
x2 = "profession"

X = (data[x1].astype(str) + data[x2].astype(str))
y = data[target]

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def save_classification(model, name, mode="a"):
    with open(f"./classification.txt", mode, encoding="utf-8") as f:
        score = ((cross_val_score(model, X, y, cv=kfold)).mean()*100).round(1)
        f.write(f"{name}\n")
        f.write(f"Tikslumas: {score}%\n")

def run_classification():
    tfidf_vectorizer = ("tfidf", TfidfVectorizer(stop_words="english",ngram_range=(1,2),lowercase=True,sublinear_tf=True))
    for alpha in [0.1, 0.5, 1.0]:
        save_classification(
            Pipeline([tfidf_vectorizer, ("clf", MultinomialNB(alpha=alpha))]),
            f"Naive Bayes, alpha={alpha}",
            mode="w" if alpha == 0.1 else "a"
        )
    with open(f"./classification.txt", "a", encoding="utf-8") as f:
            f.write(f"\n")
    for C in [0.1, 1, 10]:
        save_classification(
            Pipeline([tfidf_vectorizer, ("clf", LogisticRegression(C=C))]),
            f"Logistic Regression, C={C}"
        )
    with open(f"./classification.txt", "a", encoding="utf-8") as f:
            f.write(f"\n")
    for C in [0.1, 1, 10]:
        save_classification(
            Pipeline([tfidf_vectorizer, ("clf", LinearSVC(C=C))]),
            f"Linear SVM, C={C}"
        )

def save_clustering(model, X_vector, name, mode="a"):
    labels = model.fit_predict(X_vector)
    ads = adjusted_rand_score(y, labels)
    sil_cos = silhouette_score(X_vector, labels, metric="cosine")

    pca = PCA(n_components=50, random_state=42).fit_transform(X_vector)
    sil_cos_pca = silhouette_score(pca, labels, metric="cosine")

    with open("./clustering.txt", mode, encoding="utf-8") as f:
        f.write(f"{name}\n")
        f.write(f"Adjusted Rand Score: {ads:.3f}\n")
        f.write(f"Silhouette Score (cosine): {sil_cos:.3f}\n")
        f.write(f"Silhouette Score (cosine) with PCA: {sil_cos_pca:.3f}\n")

    plt.figure(figsize=(7, 5))
    plt.scatter(pca[:, 0], pca[:, 1], c=labels, s=20)
    plt.title(f"{name}")
    plt.tight_layout()
    plt.savefig(f"./{name}.png", dpi=150, bbox_inches="tight")
    plt.close()

def run_clustering():
    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True
    )
    X_vector = tfidf_vectorizer.fit_transform(X)
    for k in [3, 5, 6]:
        save_clustering(
            AgglomerativeClustering(n_clusters=k, linkage="ward"),
            X_vector.toarray(),
            name=f"Agglomerative, k={k}",
            mode="w" if k==3 else "a"
        )
    with open(f"./clustering.txt", "a", encoding="utf-8") as f:
            f.write(f"\n")
    for k in [3, 5, 6]:
        save_clustering(
            KMeans(n_clusters=k, n_init=20, random_state=42),
            X_vector,
            name=f"KMeans, k={k}",
            mode="a"
        )

run_classification()
run_clustering()