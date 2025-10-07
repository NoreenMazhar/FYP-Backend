FROM mysql:8.0

# Default environment variables (can be overridden at runtime with -e)
ENV MYSQL_ROOT_PASSWORD=root
ENV MYSQL_DATABASE=FYP-DB
ENV MYSQL_USER=FYP-USER
ENV MYSQL_PASSWORD=FYP-PASS 

# Copy all SQL files to the MySQL init directory. These run on first container startup
# if the data directory is empty.
COPY SQL/*.sql /docker-entrypoint-initdb.d/

# Expose default MySQL port
EXPOSE 3306

# Optional healthcheck to ensure MySQL is accepting connections
HEALTHCHECK --interval=10s --timeout=5s --retries=10 CMD mysqladmin ping -h 127.0.0.1 -uroot -p$MYSQL_ROOT_PASSWORD || exit 1

# Use UTF-8 defaults
CMD ["mysqld", "--character-set-server=utf8mb4", "--collation-server=utf8mb4_unicode_ci"]
