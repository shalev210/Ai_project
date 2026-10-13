# PostgreSQL 16 with the pgvector extension preinstalled.
# Every .sql file in /docker-entrypoint-initdb.d runs once, on first start
# of an empty data volume — this applies our schema migration automatically.
FROM pgvector/pgvector:pg16

COPY migrations/ /docker-entrypoint-initdb.d/
