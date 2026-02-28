# Flight Delay Prediction System

This is a production-oriented ML system that predicts whether a given flight will be delayed by more than 15 minutes. 

## Project Stack
- **Data Engineering**: Data extracting from BTS records, ingested via pandas/SQLAlchemy into **PostgreSQL**.
- **ML**: Scikit-Learn, LightGBM, XGBoost. Logged to **MLflow**.
- **Inference**: **FastAPI** application containerized via Docker.
- **CI/CD**: GitHub Actions workflow.
- **Deployment**: Local Docker Compose / AWS deployment scripts included.

## Getting Started Locally

1. Place the `T_ONTIME_REPORTING.csv` dataset in the root directory.
2. Ensure Docker Desktop is running.
3. Start the infrastructure (PostgreSQL + MLflow tracking server + FastAPI server):
   ```bash
   docker-compose up -d
   ```
4. Start a local shell / terminal and execute Data Engineering pipeline:
   ```bash
   pip install -r requirements.txt
   python -m etl.db_setup
   python -m etl.process_data
   ```
5. Trigger Model Training:
   ```bash
   python -m ml.train
   ```
   **Note:** The models log directly to your local MLflow instance running at `http://localhost:5000`. 
   The best run is saved locally in `models/best_run.txt`.
   
6. The FastAPI will pick up the model if available (you may need to restart the fast_api container after training to warm it up!)
   ```bash
   docker restart flight_api
   ```
   Check the API docs: `http://localhost:8000/docs`

## Cloud Deployment
Refer to `scripts/deploy.sh` to provision an AWS EC2 instance and RDS database, and `.github/workflows/main.yml` which deploys images to AWS ECR.
