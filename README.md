### CARP

**CARP** is a hospital artificial intelligence infrastructure which leverages differential privacy to improve a public AI model without hospitals revealing sensitive patient information. The project was created by Peter Liu, Charles Liu, Alex Duong, and Russell Habib for the 2026 Crimson Code Hackathon (Advanced Track).

The project consists of two main components:
1. Hospital Client (local)
2. Main AI Server (global)

The _hospital client_ allows hospitals to upload sensitive information (through CSVs) locally and train AI models with differential privacy to collect weights. Hospitals can test out the accuracy of their model through this local website. Hospitals can customize $\epsilon$, the privacy/accuracy tradeoff to decide how much they want to contribute to the global AI model on the public server. $\epsilon = 1$ is a standard reasonable amount to satisfy HIPPA. When ready, hospitals can upload their weights to the main AI server to try and improve the public model. This portion is _dockerized_ to ensure no environment issues and smooth usage in hospitals, saving time on debugging and package management.

The _main AI server_ is a public website which allows hospitals to upload weights to improve a public AI which anyone can query. For this prototype, we focus on analyzing diabetes risk. Users can enter their personal information and get a percentage analyzing their risk of diabetes.

### Impact

This project isn't just a diabetes predictor, which is a classical machine learning exercise you can find anywhere. Instead of merely reinventing the wheel, we are proving a concept and realizing an opportunity. Many fields like finance and healthcare have sensitive data which understandably they don't want to reveal. However, differential privacy offers a means to leverage this data while respecting privacy concerns. This data could be used to train AI models which can not just improve revenue, delight customers, and advance research---but save lives. Our project is more than a reinventing the wheel, its a statement: privacy-preserving computation is an underlooked tool which people can leverage for greater good.

#  How to Run

The project has two components:
- **Hospital Client** — a local Docker app for hospitals to train and upload model weights
- **Main AI Server** — a public web server where aggregated weights power a queryable AI model

---

## Prerequisites

Make sure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- [Git](https://git-scm.com/)
- [Node.js](https://nodejs.org/) (if running outside Docker)
- [Python 3.10+](https://www.python.org/) (if running outside Docker)

---

## 1. Clone the Repository

```bash
git clone https://github.com/CharlesLiuCool/carp-crimsoncode-2026.git
cd carp-crimsoncode-2026
```

---

## 2. Set Up Environment Variables

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

Open `.env` and configure any required API keys or settings (e.g., server URLs, ports).

---

## 3. Running with Docker (Recommended)

All Docker files live in the `Dockerize/` folder. From the **project root**, run:

```bash
docker compose -f Dockerize/docker-compose.yml up --build -d
```

This builds and starts all services (build context is the project root so the app code is included):
- **Hospital client** — backend on port 8000, frontend on 8081
- **Main AI server** — backend on 8001, frontend on 8082, PostgreSQL for weights

Once running, visit:
- **Hospital Client:** http://localhost:8081 (frontend), http://localhost:8000 (backend API)
- **Main AI Server:** http://localhost:8082 (frontend), http://localhost:8001 (backend API)

Ensure `.env` exists at the project root (copy from `.env.example`) if the server needs API keys (e.g. Groq/Gemini for AI guidance).

To stop all services:

```bash
docker compose -f Dockerize/docker-compose.yml down
```

---

## 4. Running Without Docker (Manual Setup)

### Backend (Server)

```bash
cd server
pip install -r ../requirements.txt
python app.py
```

### Frontend (Hospital Client)

```bash
cd hospital_client
npm install
npm start
```

---

## 5. Using the Hospital Client

1. **Upload a CSV** of patient data through the local web interface.
2. The client will **train a local model** using differential privacy.
3. Adjust **ε (epsilon)** to tune the privacy/accuracy tradeoff. A value of `ε = 1` satisfies HIPAA-level privacy.
4. **Test your local model's accuracy** before submitting.
5. When satisfied, **upload the model weights** to the main AI server to contribute to the global model.

---

## 6. Using the Main AI Server

Visit the public server URL and enter personal health information to receive a **diabetes risk percentage** powered by the aggregated model weights contributed by hospitals.(we dont have a url yet, it was expensive)

---

## 7. Running Tests

To run the secure aggregation tests:

```bash
pip install -r requirements-project.txt
python test_secure_agg.py
```

---

## Project Structure

```
carp-crimsoncode-2026/
├── hospital_client/          # Local hospital frontend + backend
├── server/                   # Public AI server frontend + backend
├── Dockerize/                # All Docker config (run from project root with -f Dockerize/docker-compose.yml)
│   ├── docker-compose.yml    # Orchestrates all services
│   ├── Dockerfile.backend    # Hospital client backend image
│   ├── Dockerfile.frontend   # Hospital client frontend image
│   ├── Dockerfile.server-backend
│   └── Dockerfile.server-frontend
├── .env.example              # Environment variable template (copy to .env at project root)
├── requirements-project.txt # Python dependencies
└── test_secure_agg.py       # Secure aggregation tests
```

---

## Built By

Peter Liu, Charles Liu, Alex Duong, and Russell Habib — WSU CrimsonCode 2026 Hackathon (Advanced Track).
