import oracledb
import os

try:
    connection = oracledb.connect(
        user=os.environ.get("DB_USER", "rag_user"),
        password=os.environ.get("DB_PASSWORD", "rag_password"),
        dsn="127.0.0.1:1521/FREEPDB1"
    )
    print("✅ Successfully connected to Oracle Database")
    connection.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")
