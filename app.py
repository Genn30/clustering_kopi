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

client = MongoClient("mongodb://localhost:27017/")
db = client["user_database"]
users_collection = db["users"]
history_collection = db["history"]

def get_cluster_categories(n):
    if n == 2: return ["Rendah", "Tinggi"]
    if n == 3: return ["Rendah", "Sedang", "Tinggi"]
    if n == 4: return ["Rendah", "Sedang", "Menengah Tinggi", "Tinggi"]
    return ["Sangat Rendah", "Rendah", "Cukup Rendah", "Sedang", "Cukup Tinggi", "Tinggi", "Sangat Tinggi"]

def extract_year_columns(df, keyword):
    return [col for col in df.columns if keyword in col and col.split()[-1].isdigit()]

def plot_bar(df, lokasi_col, fitur, colors, file_id):
    top = df.sort_values(fitur, ascending=False).head(20)
    plt.figure(figsize=(12, 6))
    plt.bar(top[lokasi_col], top[fitur], color=top['Color'])
    plt.xticks(rotation=90)
    plt.title(f"Top 20 {lokasi_col} berdasarkan {fitur}")
    plt.tight_layout()
    path = f"static/img/grafik_{fitur}_{file_id}.png"
    plt.savefig(path)
    plt.close()
    return path

def plot_trend(df, lokasi_col, cols, file_id, fitur):
    top = df.sort_values(cols[-1], ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    for _, row in top.iterrows():
        label = row[lokasi_col]
        plt.plot([int(c.split()[-1]) for c in cols], row[cols].values, label=label, marker='o')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"Tren Tahunan {fitur}")
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    path = f"static/img/tren_{fitur}_{file_id}.png"
    plt.savefig(path)
    plt.close()
    return path

def proses_clustering(uploaded_file, method, n_clusters, is_ekspor=False):
    results = []
    grafik_paths, trend_paths = {}, {}
    peta_path, dendro_path = None, None
    cluster_table, count_table = [], []

    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static/img', exist_ok=True)
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{uploaded_file.filename}"
    file_path = os.path.join('uploads', filename)
    uploaded_file.save(file_path)

    df = pd.read_excel(file_path)
    lokasi_col = 'Negara Tujuan' if is_ekspor else 'Lokasi'

    if lokasi_col not in df.columns or 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        flash("Dataset tidak valid. Pastikan format sesuai.", "error")
        return [], {}, {}, None, None, [], []

    if is_ekspor:
        berat_cols = extract_year_columns(df, 'Berat')
        value_cols = extract_year_columns(df, 'Value')
        if not berat_cols or not value_cols:
            flash("Dataset ekspor harus memiliki kolom 'Berat' dan 'Value' per tahun.", "error")
            return [], {}, {}, None, None, [], []
        df['Total_Berat'] = df[berat_cols].sum(axis=1)
        df['Total_Value'] = df[value_cols].sum(axis=1)
        fitur_cols = ['Total_Berat', 'Total_Value']
    else:
        produksi_cols = extract_year_columns(df, 'Produksi')
        luas_cols = extract_year_columns(df, 'Luas')
        produktivitas_cols = extract_year_columns(df, 'Produktivitas')
        if not produksi_cols or not luas_cols or not produktivitas_cols:
            flash("Dataset produksi harus memiliki kolom 'Produksi', 'Luas', dan 'Produktivitas' per tahun.", "error")
            return [], {}, {}, None, None, [], []
        df['Total_Produksi'] = df[produksi_cols].sum(axis=1)
        df['Total_Luas'] = df[luas_cols].sum(axis=1)
        df['Total_Produktivitas'] = df[produktivitas_cols].sum(axis=1)
        fitur_cols = ['Total_Produksi', 'Total_Luas', 'Total_Produktivitas']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[fitur_cols])

    start = time.time()
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto') if method == "kmeans" else AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X_scaled)
    algo = "K-Means" if method == "kmeans" else "Agglomerative"
    exec_time = round(time.time() - start, 4)
    silhouette = round(silhouette_score(X_scaled, labels), 2)
    dbi = round(davies_bouldin_score(X_scaled, labels), 2)

    df['Cluster'] = labels
    colormap = cm.get_cmap('tab10', n_clusters)
    cluster_colors = [mcolors.rgb2hex(colormap(i)) for i in range(n_clusters)]
    df['Color'] = df['Cluster'].map(lambda c: cluster_colors[c])

    utama = fitur_cols[0]
    cluster_means = df.groupby('Cluster')[utama].mean().sort_values()
    kategori = get_cluster_categories(n_clusters)
    mapping = {cluster: kategori[i] for i, cluster in enumerate(cluster_means.index)}
    df['Kategori'] = df['Cluster'].map(mapping)

    cluster_table = [[f"Cluster {i}", ', '.join(df[df['Cluster'] == i][lokasi_col].tolist())] for i in range(n_clusters)]
    count_table = [[f"Cluster {i}", int((df['Cluster'] == i).sum())] for i in range(n_clusters)]

    # === Simpan grafik batang dan tren
    for fitur in fitur_cols:
        grafik_paths[fitur] = plot_bar(df, lokasi_col, fitur, cluster_colors, file_id)

    if is_ekspor:
        trend_paths = {
            'Berat': plot_trend(df, lokasi_col, berat_cols, file_id, 'Berat'),
            'Value': plot_trend(df, lokasi_col, value_cols, file_id, 'Value')
        }
    else:
        trend_paths = {
            'Produksi': plot_trend(df, lokasi_col, produksi_cols, file_id, 'Produksi'),
            'Luas': plot_trend(df, lokasi_col, luas_cols, file_id, 'Luas'),
            'Produktivitas': plot_trend(df, lokasi_col, produktivitas_cols, file_id, 'Produktivitas')
        }

    # === Peta Interaktif
    peta = folium.Map(location=[-2, 117], zoom_start=3 if not is_ekspor else 2)
    for _, row in df.iterrows():
        popup = f"<b>{row[lokasi_col]}</b><br>Cluster: {row['Cluster']}<br>Kategori: {row['Kategori']}<br>"
        for fitur in fitur_cols:
            popup += f"{fitur.replace('_', ' ')}: {int(row[fitur]):,}<br>"
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=6, color=row['Color'], fill=True, fill_opacity=0.7, popup=popup
        ).add_to(peta)

    legend = '<div style="position: fixed; bottom: 20px; left: 20px; z-index:9999; background-color:white; padding: 10px; border:2px solid grey;"><strong>Legenda Cluster:</strong><br>'
    for i in range(n_clusters):
        legend += f'<i style="background:{cluster_colors[i]}; width:10px; height:10px; display:inline-block;"></i> Cluster {i} - {mapping[i]}<br>'
    legend += '</div>'
    peta.get_root().html.add_child(folium.Element(legend))
    peta_path = f"static/img/peta_{file_id}.html"
    peta.save(peta_path)

    if method == "agglo":
        plt.figure(figsize=(15, 7))
        linked = linkage(X_scaled, method='ward')
        threshold = linked[-(n_clusters - 1), 2]
        dendrogram(linked, color_threshold=threshold, orientation='top', show_leaf_counts=True)
        plt.tight_layout()
        dendro_path = f"static/img/dendrogram_{file_id}.png"
        plt.savefig(dendro_path)
        plt.close()

    history_collection.insert_one({
        "username": session.get("username", "guest"),
        "timestamp": datetime.now(),
        "type": "Ekspor" if is_ekspor else "Produksi",
        "method": algo,
        "n_clusters": n_clusters,
        "silhouette": silhouette,
        "dbi": dbi,
        "time_exec": exec_time,
        "peta_path": peta_path,
        "dendro_path": dendro_path,
        "cluster_table": cluster_table,
        "count_table": count_table,
        "grafik_paths": list(grafik_paths.values()),
        "trend_paths": list(trend_paths.values())
    })

    results.append({
        'Algorithm': algo,
        'Clusters Num.': n_clusters,
        'Avg. Silhouette': silhouette,
        'Davies Bouldin Index': dbi,
        'Execution Time (s)': exec_time
    })

    return results, grafik_paths, trend_paths, peta_path, dendro_path, cluster_table, count_table

@app.route('/')
def home(): return render_template('home.html')

@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    hasil = [], {}, {}, None, None, [], []
    if request.method == 'POST':
        if 'username' not in session: flash("Anda harus login terlebih dahulu.", "error")
        else:
            f = request.files.get('file')
            method = request.form.get('algorithm')
            n = int(request.form.get('n_clusters', 3))
            if f and method: hasil = proses_clustering(f, method, n, is_ekspor=False)
            else: flash("Lengkapi semua pilihan clustering.", "error")
    return render_template('clustering.html', results=hasil[0], grafik_paths=hasil[1], trend_paths=hasil[2],
        peta_path=hasil[3], dendro_path=hasil[4], cluster_table=hasil[5], count_table=hasil[6])

@app.route('/clustering/ekspor', methods=['GET', 'POST'])
def clustering_ekspor():
    hasil = [], {}, {}, None, None, [], []
    if request.method == 'POST':
        if 'username' not in session: flash("Anda harus login terlebih dahulu.", "error")
        else:
            f = request.files.get('file')
            method = request.form.get('method')
            n = int(request.form.get('n_clusters', 3))
            if f and method: hasil = proses_clustering(f, method, n, is_ekspor=True)
            else: flash("Lengkapi semua pilihan clustering.", "error")
    return render_template('clustering_ekspor.html', results=hasil[0], grafik_paths=hasil[1], trend_paths=hasil[2],
        peta_path=hasil[3], dendro_path=hasil[4], cluster_table=hasil[5], count_table=hasil[6])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        if users_collection.find_one({"username": uname, "password": pwd}):
            session['username'] = uname
            flash("Login berhasil.", "success")
            return redirect(url_for('home'))
        flash("Username atau password salah.", "error")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        if users_collection.find_one({"username": uname}): flash("Username sudah digunakan.", "error")
        else:
            users_collection.insert_one({"username": uname, "password": pwd})
            flash("Registrasi berhasil. Silakan login.", "success")
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Anda telah logout.", "info")
    return redirect(url_for('home'))

@app.route('/history')
def history():
    all_history = list(history_collection.find().sort("timestamp", -1))
    grouped = defaultdict(list)
    for h in all_history:
        tgl = h["timestamp"].strftime('%d/%m/%Y')
        grouped[tgl].append(h)
    return render_template('history.html', grouped_history=grouped)

@app.route('/about')
def about(): return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
