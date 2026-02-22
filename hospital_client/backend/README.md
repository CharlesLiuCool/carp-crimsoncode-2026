# Description

Runs the backend functionality of the hospital client. Backend takes in csv from frontend and trains a model locally and outputs weights using differential privacy. The weights are given a random "mask" value from the central server which then gets added. The masked number then gets canceled out on the central server after 3 models which helps obscures hospital wide data.

# To Run

1. Navigate to hospital_client/backend
```
cd hospital_client/backend
```
2. If first time running
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../../requirements-project.txt
```
3. If not first time running
```
source .venv/bin/activate
```
4. Start the backend
```
python3 main.py
```
