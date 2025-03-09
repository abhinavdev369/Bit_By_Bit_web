from pymongo import MongoClient

# Replace with your actual connection string
MONGO_URI = "mongodb+srv://flaskuser9249:9249@flask1.w9mtn.mongodb.net/?retryWrites=true&w=majority&appName=Flask1"

try:
    client = MongoClient(MONGO_URI)
    db = client.list_database_names()
    print("Connected to MongoDB! Databases:", db)
except Exception as e:
    print("Error:", e)