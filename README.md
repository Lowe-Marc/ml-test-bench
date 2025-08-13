# Installation
```
brew install python@3.10
python3.10 -m venv .venv
source .venv/bin/activate


brew install cmake
brew install apache-arrow
export CMAKE_PREFIX_PATH="$(brew --prefix apache-arrow)/lib/cmake"
pip install pyarrow


pip install -r requirements.txt
pip install -e .
```


# Phase 1 – Core ML Prototype (approx. 4–5 days)
### Simulate or Ingest Sample Data (8 h)
Goal: Stand up a realistic data source so you can iterate end-to-end without waiting on production feeds.

- [ ] Determine data schema
- [ ] Write generator
    - Time series
    - Include anomaly spikes
    - Write to parquet

### Exploratory Data Analysis & Feature Engineering (12 h)
Goal: Understand the characteristics of your anomalies and craft features for your model.
Subtasks:

- [ ] Plot sample data with matplotlib
- [ ] Identify features to engineer
    - Rolling means
    - Diffs
    - Z-scores
    - Autocrrelations
    - Outlier patterns
- [ ] Scale/normalize features and save transformers?
    - scikit-learn StandardScaler?
- [ ] Tests for feature code

### Prototype & Train Anomaly Model (16 h)
Goal: Get a first-cut model that flags anomalies reliably.
Subtasks:

- [ ] Choose baseline algorithm (isolation forest, autoencoder)
- [ ] Training script
    - Accepts params for data, hyperparameters
    - Log metrics (precision/recal, ROC)
- [ ] Implement hyperparameter sweep?
- [ ] Version and serialize final model + scaler
    - .pkl or ONNX?


### Evaluate & Threshold Selection (12 h)
Goal: Establish your anomaly-score cutoff and visualize performance.
Subtasks:
- [ ] Compute scores for "validation" data set
- [ ] Plot score distributions for normal vs anomaly
- [ ] Sweep thresholds to find optimal F1 or target false-alarm rate (?)
- [ ] Write a CLI tool to apply threshold & output flagged timestamps/IDs (?)


# Phase 2 – Local Deployment & Packaging (2–3 days)
### Build Inference API (6 h)
Goal: Wrap your model in a REST endpoint so you can call it from anywhere.
Subtasks:
- [ ] Scaffold a FastAPI (or Flask) app with /predict POST endpoint (2 h)
    - Load serialized model & scaler on startup (1 h)
- [ ] Parse incoming JSON, apply feature pipeline + model, return anomaly scores (2 h)
- [ ] Add basic input validation + error handling (1 h)

### Containerize with Docker (4 h)
Goal: Package your API, data-gen, and any helper scripts into a single Docker image.
Subtasks:
- [ ] Write a multi-stage Dockerfile (builder + slim runtime) (2 h)
- [ ] Add health-check and default command to run the API server (1 h)
- [ ] Build and run locally to smoke-test (1 h)

7. Orchestrate Locally with Docker Compose (6 h)
Goal: Stand up your entire stack (data-gen → API) in one command.
Subtasks:

Create a docker-compose.yml with services for data-gen script & API (2 h)

Mount volumes so you can iterate code without rebuilding containers (1 h)

Configure networks & ports; test end-to-end (2 h)

Document local bring-up steps in README (1 h)

# Phase 3 – Testing, CI/CD & Monitoring (2–3 days)
8. Automated Testing & Linting (8 h)
Goal: Ensure code quality and catch regressions before merging.
Subtasks:

Write Pytest unit tests for data-gen, feature transforms, threshold CLI (4 h)

Add linting (Flake8 or Black) and type-checking (mypy) (2 h)

Configure pre-commit hooks for formatting + tests (1 h)

Document “how to test” in README (1 h)

9. Basic CI Pipeline (GitHub Actions) (8 h)
Goal: Automatically build, test, and lint on every push/PR.
Subtasks:

Create a GitHub Actions workflow YAML (jobs: build, test, lint) (3 h)

Add a step to build the Docker image & optionally push to a test registry (2 h)

Enable branch protection requiring CI pass before merge (1 h)

Verify workflow runs and fix failures (2 h)

10. Add Logging & Metrics (8 h)
Goal: Capture inference requests/responses and track volumes + latency.
Subtasks:

Integrate Python logging (structured JSON format) (2 h)

Expose Prometheus metrics via an endpoint (/metrics) (2 h)

Run a local Prometheus + Grafana stack via Docker Compose (2 h)

Create basic Grafana dashboard for request rate, error rate (2 h)

11. Data Drift Detection Module (12 h)
Goal: Monitor incoming data distribution and alert on drift.
Subtasks:

Integrate EvidentlyAI or write simple PSI / KS-test checks (4 h)

Schedule a periodic job (cron-style) to snapshot recent feature stats (2 h)

Compare to reference distribution and log/emit metric on drift score (4 h)

Add Grafana alerting rule or simple email/webhook notification (2 h)

# Phase 4 – Cloud-Ready & Lift-and-Shift (2–3 days)
12. Parameterize for Cloud (6 h)
Goal: Externalize configs so you can swap local volumes for S3, secrets, etc.
Subtasks:

Replace hard-coded paths with env vars (12-factor) (2 h)

Add support for S3-backed model artifacts & sample data (2 h)

Integrate a secrets manager stub (e.g. AWS Secrets Manager local/mock) (2 h)

13. Infrastructure as Code (Terraform) (10 h)
Goal: Define your cloud infra in code for repeatable deployments.
Subtasks:

Write Terraform modules for ECS/EKS (or GKE/AKS) cluster (4 h)

Provision container registry (ECR/GCR/ACR) and attach policies (2 h)

Create IAM roles, security groups, service accounts (2 h)

Validate with terraform plan & apply in a sandbox account (2 h)

14. Kubernetes Deployment & Helm (12 h)
Goal: Deploy your containers into a managed k8s cluster with Helm charts.
Subtasks:

Scaffold a Helm chart for your API (templates for Deployment, Service, Ingress) (4 h)

Add ConfigMap/Secret objects for config & credentials (2 h)

Define Horizontal Pod Autoscaler based on Prometheus metrics (2 h)

Test deploy to dev cluster (minikube or cloud) & validate endpoints (4 h)

Total Estimated Effort
~72 – 90 hours (roughly 3–4 weeks at 20 h/week)
You can (and should) overlap tasks where sensible—e.g., start CI while finishing containerization.

Next Steps:

Pick a sprint cadence (e.g., 1 week sprints) and assign these tasks to cards.

Kick off with Phase 1, aiming to have a working model + threshold in 1 week.

Iterate with daily check-ins on progress and blockers.