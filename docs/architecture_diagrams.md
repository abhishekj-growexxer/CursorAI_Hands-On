# System Architecture Diagrams

## 1. Overall System Architecture
```mermaid
flowchart TD
    subgraph DataIngestion["Data Ingestion Layer"]
        CSV["CSV Files"] --> DL["Data Loader"]
        DL --> DV["Data Validation"]
        DV --> FE["Feature Engineering"]
        FE --> Cache["Feature Store Cache"]
    end

    subgraph ModelLayer["Model Layer"]
        Cache --> Split["Train-Test Split"]
        Split --> |Training| Train["Training Pipeline"]
        Split --> |Testing| Eval["Evaluation Pipeline"]
        
        subgraph Models["Model Zoo"]
            direction LR
            ARIMA["ARIMA Model"]
            Prophet["Prophet Model"]
            LSTM["LSTM Model"]
        end
        
        Train --> Models
        Models --> Eval
    end

    subgraph MLOps["MLOps Layer"]
        Eval --> |Metrics| MLflow["MLflow Tracking"]
        MLflow --> |"Best Model"| Registry["Model Registry"]
        Registry --> Serving["Model Serving"]
    end

    subgraph Orchestration["Orchestration Layer"]
        Prefect["Prefect Scheduler"] --> |Trigger| DataIngestion
        Prefect --> |Monitor| MLOps
        Prefect --> |Schedule| Retrain["Retraining Pipeline"]
    end

    subgraph Monitoring["Monitoring Layer"]
        Serving --> Metrics["Performance Metrics"]
        Metrics --> Dashboard["Monitoring Dashboard"]
        Metrics --> Alerts["Alert System"]
    end
```

## 2. Data Processing Pipeline
```mermaid
flowchart LR
    subgraph Input["Input Processing"]
        CSV["CSV Source"] --> Parser["CSV Parser"]
        Parser --> Validator["Data Validator"]
        Validator --> Clean["Data Cleaning"]
    end

    subgraph Features["Feature Engineering"]
        Clean --> Time["Time Features"]
        Clean --> Lag["Lag Features"]
        Clean --> Stats["Statistical Features"]
        
        Time --> Norm["Feature Normalization"]
        Lag --> Norm
        Stats --> Norm
    end

    subgraph Quality["Quality Checks"]
        Norm --> Missing["Missing Values"]
        Norm --> Outliers["Outlier Detection"]
        Norm --> Drift["Drift Detection"]
        
        Missing --> QReport["Quality Report"]
        Outliers --> QReport
        Drift --> QReport
    end

    subgraph Storage["Data Storage"]
        QReport --> |Valid| Store["Feature Store"]
        QReport --> |Invalid| Error["Error Handler"]
        Store --> Cache["Cache Layer"]
    end
```

## 3. Model Training Workflow
```mermaid
flowchart TD
    subgraph DataPrep["Data Preparation"]
        FS["Feature Store"] --> Split["Time-based Split"]
        Split --> |Training| Train["Training Set"]
        Split --> |Validation| Val["Validation Set"]
        Split --> |Test| Test["Test Set"]
    end

    subgraph Training["Model Training"]
        Train --> |Config| HP["Hyperparameters"]
        HP --> |ARIMA| ARIMA["ARIMA Training"]
        HP --> |Prophet| Prophet["Prophet Training"]
        HP --> |LSTM| LSTM["LSTM Training"]
    end

    subgraph Evaluation["Model Evaluation"]
        ARIMA --> |Predict| Metrics["Metrics Calculation"]
        Prophet --> |Predict| Metrics
        LSTM --> |Predict| Metrics
        Val --> Metrics
        
        Metrics --> |RMSE| Scores["Score Board"]
        Metrics --> |MAE| Scores
        Metrics --> |MAPE| Scores
    end

    subgraph Selection["Model Selection"]
        Scores --> Rank["Model Ranking"]
        Rank --> Best["Best Model Selection"]
        Best --> Final["Final Evaluation"]
        Test --> Final
    end
```

## 4. MLOps Pipeline
```mermaid
flowchart TD
    subgraph Schedule["Scheduling"]
        Cron["Cron Trigger"] --> Flow["Prefect Flow"]
        Flow --> Check["Status Check"]
    end

    subgraph Training["Training Pipeline"]
        Check --> |New Data| Prep["Data Preparation"]
        Prep --> Train["Model Training"]
        Train --> Eval["Model Evaluation"]
    end

    subgraph Registry["Model Registry"]
        Eval --> |Metrics| Compare["Performance Compare"]
        Compare --> |Better| Register["Register Model"]
        Compare --> |Worse| Keep["Keep Current"]
        
        Register --> Version["Version Control"]
        Version --> Deploy["Deployment"]
    end

    subgraph Monitor["Monitoring"]
        Deploy --> Track["Performance Tracking"]
        Track --> Drift["Drift Detection"]
        Track --> Health["Health Checks"]
        
        Drift --> |Alert| Action["Action Required"]
        Health --> |Alert| Action
    end
```

## 5. Monitoring System
```mermaid
flowchart LR
    subgraph DataMonitor["Data Monitoring"]
        Input["Input Data"] --> Quality["Quality Metrics"]
        Input --> Dist["Distribution Analysis"]
        Input --> Missing["Missing Patterns"]
    end

    subgraph ModelMonitor["Model Monitoring"]
        Pred["Predictions"] --> Acc["Accuracy Metrics"]
        Pred --> Latency["Response Time"]
        Pred --> Error["Error Patterns"]
    end

    subgraph SysMonitor["System Monitoring"]
        Res["Resources"] --> CPU["CPU Usage"]
        Res --> Mem["Memory Usage"]
        Res --> Disk["Storage Usage"]
    end

    subgraph Alerts["Alert System"]
        Quality --> Rules["Alert Rules"]
        Acc --> Rules
        CPU --> Rules
        
        Rules --> Notify["Notifications"]
        Rules --> Action["Action Items"]
    end

    subgraph Dashboard["Monitoring Dashboard"]
        Quality --> Viz["Visualizations"]
        Acc --> Viz
        CPU --> Viz
        
        Viz --> Report["Reports"]
        Report --> Export["Exportable Formats"]
    end
```

## 6. Retraining Decision Flow
```mermaid
flowchart TD
    subgraph Triggers["Retraining Triggers"]
        Time["Time-based"] --> Check["Check Conditions"]
        Perf["Performance Drop"] --> Check
        Drift["Data Drift"] --> Check
    end

    subgraph Decision["Decision Making"]
        Check --> Analyze["Analysis"]
        Analyze --> |Threshold Met| Yes["Initiate Retraining"]
        Analyze --> |Threshold Not Met| No["Skip Retraining"]
    end

    subgraph Action["Actions"]
        Yes --> Notify["Notify Stakeholders"]
        Yes --> Log["Log Decision"]
        Yes --> Train["Start Training"]
        
        No --> Monitor["Continue Monitoring"]
        No --> Log
    end

    subgraph Validation["Post-Retraining"]
        Train --> Eval["Evaluate New Model"]
        Eval --> |Better| Deploy["Deploy Model"]
        Eval --> |Worse| Rollback["Rollback"]
    end
``` 