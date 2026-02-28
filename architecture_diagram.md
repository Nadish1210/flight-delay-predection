# System Architecture Diagram

```mermaid
flowchart TD
    %% Define styles
    classDef rawData fill:#e1f5fe,stroke:#333,stroke-width:1px
    classDef etl fill:#fff3e0,stroke:#333,stroke-width:1px
    classDef db fill:#e8f5e9,stroke:#333,stroke-width:1px
    classDef ml fill:#f3e5f5,stroke:#333,stroke-width:1px
    classDef api fill:#ffebee,stroke:#333,stroke-width:1px
    classDef cloud fill:#e0f7fa,stroke:#333,stroke-width:1px

    %% Components
    A(Raw CSV Data<br/>BTS On-Time Dataset) ::: rawData
    
    subgraph Data Engineering Pipeline
        B(ETL Script<br/>Chunked Ingestion & Cleaning) ::: etl
    end
    
    C[(PostgreSQL DB<br/>Flight Delays Table)] ::: db
    
    subgraph Machine Learning Pipeline
        D{Feature Engineering} ::: ml
        E(Model Training<br/>LogReg / LightGBM / XGBoost) ::: ml
        F[(MLflow Model Registry<br/>Experiment Tracking)] ::: ml
    end
    
    subgraph Inference & API Serving
        G[FastAPI Server<br/>/predict endpoint] ::: api
        H(Data Drift Check<br/>Evidently AI Mock) ::: api
        I[(Inference Logs<br/>JSONL)] ::: api
    end
    
    subgraph Cloud Deployment
        J(Docker Image) ::: cloud
        K(AWS ECR) ::: cloud
        L(AWS EC2 Instance) ::: cloud
    end
    
    subgraph CI/CD
        M(GitHub Actions<br/>Lint, Test, Build, Push) ::: cloud
    end

    %% Connections
    A -->|Extract| B
    B -->|Transform & Load| C
    C -->|SQL Query| D
    D --> E
    E -->|Log Metrics & Artifacts| F
    
    F -.->|Load Best Model| G
    G -->|Monitor| H
    G -->|Store Logs| I
    
    G --> J
    J -->|Push| K
    K -->|Pull & Run| L
    M -->|Trigger Deploy| K
    
    User((Client Request)) -->|POST /predict| L
```

## Description
1. **Raw CSV Data** is processed by Python `pandas` in chunks to manage memory footprint.
2. Cleaned data is ingested directly into a **PostgreSQL** database using `SQLAlchemy`.
3. The **ML Training Pipeline** connects to PostgreSQL, reads data, engineers features, and fits LightGBM, XGBoost, and Logistic Regression models.
4. **MLflow** tracks hyperparameters, scores (ROC-AUC, Accuracy), and stores the finalized `pyfunc` model artifacts.
5. The **FastAPI** web application loads the best model out of the registry and provides a `/predict` endpoint.
6. The entire stack is containerized locally with **Docker Compose**, and **GitHub Actions** manages CI/CD to push built images to **AWS ECR** for subsequent **AWS EC2** deployments.
