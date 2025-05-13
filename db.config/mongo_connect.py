# from pymongo import MongoClient

# # URI
# MONGO_URI = "mongodb+srv://ar123:test12345@dbcluster.u7vbzvk.mongodb.net/?retryWrites=true&w=majority&appName=DbCluster"

# # Koneksi ke MongoDB
# client = MongoClient(MONGO_URI)

# # Tes akses ke database dan collection
# try:
#     db = client["db_uji"]  # bisa ganti nama database
#     col = db["collection_uji"]
#     col.insert_one({"status": "terhubung"})
#     print("✅ Berhasil terkoneksi ke MongoDB Atlas dan menulis dokumen.")
# except Exception as e:
#     print("❌ Gagal terkoneksi:", e)

# from pymongo import MongoClient

# client = MongoClient("mongodb://localhost:27017/")
# db = client["user_database"]
# users_collection = db["users"]
# history_collection = db["history"]

from pymongo import MongoClient

# URL koneksi MongoDB 
MONGO_URI = "mongodb://localhost:27017/"

try:
   
    client = MongoClient(MONGO_URI)

   
    db = client["user_database"]

    users_collection = db["users"]
    history_collection = db["history"]

    print("✅ Berhasil terhubung ke MongoDB lokal.")
except Exception as e:
    print(f"❌ Gagal terhubung ke MongoDB: {e}")
