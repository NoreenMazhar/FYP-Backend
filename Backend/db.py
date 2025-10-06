import os
import threading
import mysql.connector
from mysql.connector import pooling


class Database:
	_instance = None
	_lock = threading.Lock()

	def __init__(self) -> None:
		config = {
			"host": os.getenv("DB_HOST", "localhost"),
			"port": int(os.getenv("DB_PORT", "3306")),
			"user": os.getenv("DB_USER", os.getenv("MYSQL_USER", "FYP-USER")),
			"password": os.getenv("DB_PASSWORD", os.getenv("MYSQL_PASSWORD", "FYP-PASS")),
			"database": os.getenv("DB_NAME", os.getenv("MYSQL_DATABASE", "FYP-DB")),
			"charset": "utf8mb4",
			"collation": "utf8mb4_unicode_ci",
		}
		self._pool = pooling.MySQLConnectionPool(pool_name="fyp_pool", pool_size=5, **config)

	@classmethod
	def get_instance(cls) -> "Database":
		if cls._instance is None:
			with cls._lock:
				if cls._instance is None:
					cls._instance = Database()
		return cls._instance

	def get_connection(self):
		return self._pool.get_connection()

	def execute(self, query: str, params: tuple | None = None):
		conn = self.get_connection()
		try:
			with conn.cursor(dictionary=True) as cursor:
				cursor.execute(query, params or ())
				if cursor.with_rows:
					rows = cursor.fetchall()
					conn.commit()
					return rows
				conn.commit()
				return None
		finally:
			conn.close()


