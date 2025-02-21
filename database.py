import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get database credentials from environment variables
DB_USER = os.getenv("PG_USER")
DB_PASS = os.getenv("PG_PASS")
DB_HOST = os.getenv("PG_HOST")
DB_PORT = os.getenv("PG_PORT")
DB_NAME = "postgres"  # Default database


conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS,
    host=DB_HOST,
    port=DB_PORT
)
conn.autocommit = True

cursor = conn.cursor()

# Create a new database
cursor.execute("CREATE DATABASE flaskapp_db")

print("Database created successfully!")

cursor.close()
conn.close()
