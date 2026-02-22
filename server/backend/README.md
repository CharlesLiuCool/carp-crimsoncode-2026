
# To Run

1. Navigate to server/backend
```
cd server/backend
```
2. If first time running
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../../requirements-project.txt
```
2a. Install PostgreSQL locally and create database and user
```
psql postgres

CREATE USER carp WITH PASSWORD 'carp';
CREATE DATABASE carp OWNER carp;
GRANT ALL PRIVILEGES ON DATABASE carp TO carp;
\q
```
2b. Add .env with API key for Groq and Gemini, and PostgreSQL
3. If not first time running
```
source .venv/bin/activate
```
4. Start the backend
```
python3 main.py
```
