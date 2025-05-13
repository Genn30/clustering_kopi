import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, redirect, url_for, request, flash, session
from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import time
import folium
import uuid
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)
app.secret_key = 'secret-key'

# === MongoDB ===
client = MongoClient("mongodb://localhost:27017/")
db = client["user_database"]
users_collection = db["users"]
history_collection = db["history"]

# === Fungsi untuk menentukan kategori berdasarkan rata-rata total per cluster ===
def get_cluster_categories(n):
    if n == 2: return ["Rendah", "Tinggi"]
    if n == 3: return ["Rendah", "Sedang", "Tinggi"]
    if n == 4: return ["Rendah", "Sedang", "Menengah Tinggi", "Tinggi"]
    if n == 5: return ["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]
    if n == 6: return ["Sangat Rendah", "Rendah", "Sedikit", "Sedang", "Tinggi", "Sangat Tinggi"]
    return ["Sangat Rendah", "Rendah", "Cukup Rendah", "Sedang", "Cukup Tinggi", "Tinggi", "Sangat Tinggi"]

# === Fungsi Proses Clustering ===
def proses_clustering(uploaded_file, method, n_clusters, is_ekspor=False):
    results = []
    grafik_path, trend_path, peta_path, dendro_path = None, None, None, None
    cluster_table, count_table = [], []

    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static/img', exist_ok=True)

    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{uploaded_file.filename}"
    file_path = os.path.join('uploads', filename)
    uploaded_file.save(file_path)

    df = pd.read_excel(file_path)
    lokasi_col = 'Negara Tujuan' if is_ekspor else 'Lokasi'
    total_col = 'Berat' if is_ekspor else 'Produksi'

    if lokasi_col not in df.columns or not any(total_col in c for c in df.columns):
        flash("Dataset tidak valid. Pastikan format file sesuai template yang diberikan.", "error")
        return [], None, None, None, None, [], []

    tahun_cols = [col for col in df.columns if total_col in col and any(str(y) in col for y in range(2015, 2023))]
    if len(tahun_cols) == 0:
        flash("Dataset tidak valid. Kolom tahun tidak ditemukan.", "error")
        return [], None, None, None, None, [], []

    df['Total'] = df[tahun_cols].sum(axis=1)
    df_numeric = df[tahun_cols].copy().dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    start = time.time()
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        algo = "K-Means"
        labels = model.fit_predict(X_scaled)
    elif method == "agglo":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        algo = "Agglomerative"
        labels = model.fit_predict(X_scaled)
        linked = linkage(X_scaled, method='ward')
        threshold = linked[-(n_clusters - 1), 2]
        plt.figure(figsize=(15, 7))
        dendrogram(linked, color_threshold=threshold, orientation='top', distance_sort='descending', show_leaf_counts=True)
        plt.title("Dendrogram Agglomerative Clustering")
        plt.tight_layout()
        dendro_path = f"static/img/dendrogram_{file_id}.png"
        plt.savefig(dendro_path)
        plt.close()
    else:
        flash("Metode clustering tidak dikenali.", "error")
        return [], None, None, None, None, [], []

    df['Cluster'] = labels
    silhouette = round(silhouette_score(X_scaled, labels), 2)
    dbi = round(davies_bouldin_score(X_scaled, labels), 2)
    exec_time = round(time.time() - start, 4)

    colormap = cm.get_cmap('tab10', n_clusters)
    cluster_colors = [mcolors.rgb2hex(colormap(i)) for i in range(n_clusters)]
    df['Color'] = df['Cluster'].map(lambda c: cluster_colors[c])

    # === Kategori berdasarkan rata-rata total tiap cluster ===
    cluster_means = df.groupby('Cluster')['Total'].mean().sort_values()
    cluster_categories = get_cluster_categories(n_clusters)
    cluster_cat_map = {cluster: cluster_categories[i] for i, cluster in enumerate(cluster_means.index)}
    df['Kategori'] = df['Cluster'].map(cluster_cat_map)

    # === Tabel Anggota & Jumlah ===
    cluster_table_df = df[[lokasi_col, 'Cluster']].sort_values('Cluster')
    cluster_table = []
    for i in range(n_clusters):
        anggota = cluster_table_df[cluster_table_df['Cluster'] == i][lokasi_col].tolist()
        cluster_table.append([f"Cluster {i}", ', '.join(anggota)])
    cluster_counts = df['Cluster'].value_counts().sort_index()
    count_table = [[f"Cluster {i}", int(cluster_counts[i])] for i in range(n_clusters)]

    # === Grafik Batang Top 20 ===
    top = df.sort_values('Total', ascending=False).head(20)
    plt.figure(figsize=(12, 6))
    plt.bar(top[lokasi_col], top['Total'], color=top['Color'])
    plt.xticks(rotation=90)
    plt.title(f"Top 20 {'Negara' if is_ekspor else 'Kabupaten/Kota'} Berdasarkan Total {'Ekspor' if is_ekspor else 'Produksi'} ({algo})")
    plt.xlabel("Negara" if is_ekspor else "Kabupaten/Kota")
    plt.ylabel("Total (kg)")
    plt.tight_layout()
    grafik_path = f"static/img/grafik_{file_id}.png"
    plt.savefig(grafik_path)
    plt.close()

    # === Grafik Tren Tahunan Top 10 ===
    top10 = df.sort_values('Total', ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    for _, row in top10.iterrows():
        plt.plot([int(col.split()[-1]) for col in tahun_cols], row[tahun_cols].values, marker='o', label=row[lokasi_col])
    plt.title(f"Grafik Tren Tahunan {'Berat' if is_ekspor else 'Produksi'} (Top 10)")
    plt.xlabel("Tahun")
    plt.ylabel("Total (kg)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.75, 1])  # untuk memberi ruang bagi legend
    plt.tight_layout()
    trend_path = f"static/img/tren_{file_id}.png"
    plt.savefig(trend_path)
    plt.close()

    # === Peta Interaktif ===
    peta_path = None
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        peta = folium.Map(location=[0, 120], zoom_start=3)
        for _, row in df.iterrows():
            popup = f"<b>{row[lokasi_col]}</b><br>Cluster: {row['Cluster']}<br>Total: {int(row['Total']):,} kg<br>Kategori: {row['Kategori']}"
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=6,
                color=row['Color'],
                fill=True,
                fill_opacity=0.7,
                popup=popup
            ).add_to(peta)

        # Tambahkan legenda
        legend_html = '<div style="position: fixed; bottom: 20px; left: 20px; z-index:9999; background-color:white; padding: 10px; border:2px solid grey; border-radius:5px;"><strong>Legenda Cluster:</strong><br>'
        for i in range(n_clusters):
            legend_html += f'<i style="background:{cluster_colors[i]}; width:10px; height:10px; display:inline-block;"></i> Cluster {i} - {cluster_cat_map[i]}<br>'
        legend_html += '</div>'
        peta.get_root().html.add_child(folium.Element(legend_html))
        peta_path = f"static/img/peta_{file_id}.html"
        peta.save(peta_path)

    # Simpan ke MongoDB
    history_collection.insert_one({
        "username": session.get("username", "guest"),
        "timestamp": datetime.now(),
        "type": "Ekspor" if is_ekspor else "Produksi",
        "method": algo,
        "n_clusters": n_clusters,
        "silhouette": silhouette,
        "dbi": dbi,
        "time_exec": exec_time,
        "grafik_path": grafik_path,
        "trend_path": trend_path,
        "peta_path": peta_path,
        "dendro_path": dendro_path
    })

    results.append({
        'Algorithm': algo,
        'Clusters Num.': n_clusters,
        'Avg. Silhouette': silhouette,
        'Davies Bouldin Index': dbi,
        'Execution Time (s)': exec_time
    })

    return results, grafik_path, trend_path, peta_path, dendro_path, cluster_table, count_table

# === ROUTES ===
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({"username": username, "password": password})
        if user:
            session['username'] = username
            flash("Login berhasil.", "success")
            return redirect(url_for('home'))
        flash("Username atau password salah.", "error")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if users_collection.find_one({"username": username}):
            flash("Username sudah digunakan.", "error")
        else:
            users_collection.insert_one({"username": username, "password": password})
            flash("Registrasi berhasil. Silakan login.", "success")
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Anda telah logout.", "info")
    return redirect(url_for('home'))

@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    results, grafik_path, trend_path, peta_path, dendro_path, cluster_table, count_table = [], None, None, None, None, [], []
    if request.method == 'POST':
        if 'username' not in session:
            flash("Anda harus login terlebih dahulu.", "error")
        else:
            uploaded_file = request.files.get('file')
            method = request.form.get('algorithm')
            n_clusters = int(request.form.get('n_clusters', 3))
            if uploaded_file and method:
                results, grafik_path, trend_path, peta_path, dendro_path, cluster_table, count_table = proses_clustering(uploaded_file, method, n_clusters, is_ekspor=False)
            else:
                flash("Lengkapi semua pilihan clustering.", "error")
    return render_template('clustering.html', results=results, grafik_path=grafik_path, trend_path=trend_path,
                           peta_path=peta_path, dendro_path=dendro_path,
                           cluster_table=cluster_table, count_table=count_table)

@app.route('/clustering/ekspor', methods=['GET', 'POST'])
def clustering_ekspor():
    results, grafik_path, trend_path, peta_path, dendro_path, cluster_table, count_table = [], None, None, None, None, [], []
    if request.method == 'POST':
        if 'username' not in session:
            flash("Anda harus login terlebih dahulu.", "error")
        else:
            uploaded_file = request.files.get('file')
            method = request.form.get('method')
            n_clusters = int(request.form.get('n_clusters', 3))
            if uploaded_file and method:
                results, grafik_path, trend_path, peta_path, dendro_path, cluster_table, count_table = proses_clustering(uploaded_file, method, n_clusters, is_ekspor=True)
            else:
                flash("Lengkapi semua pilihan clustering.", "error")
    return render_template('clustering_ekspor.html', results=results, grafik_path=grafik_path, trend_path=trend_path,
                           peta_path=peta_path, dendro_path=dendro_path,
                           cluster_table=cluster_table, count_table=count_table)

@app.route('/history')
def history():
    all_history = list(history_collection.find().sort("timestamp", -1))
    grouped_history = defaultdict(list)
    for h in all_history:
        date_str = h["timestamp"].strftime('%d/%m/%Y')
        grouped_history[date_str].append(h)
    return render_template('history.html', grouped_history=grouped_history)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
