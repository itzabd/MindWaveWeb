import psycopg2

# Connect to PostgreSQL server
conn = psycopg2.connect(
    dbname="postgres",  # Default database
    user="postgres",
    password="4297",
    host="localhost",
    port="5432"
)
conn.autocommit = True

cursor = conn.cursor()

# Create a new database
cursor.execute("CREATE DATABASE flaskapp_db")

print("Database created successfully!")

cursor.close()
conn.close()
