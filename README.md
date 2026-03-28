Optimal Pit Stop Prediction in Formula 1

A GRU‑based deep learning model for predicting pit‑stop timing using 2024 F1 telemetry data.


1. Problem Statement
   
Pit‑stop timing is one of the most critical strategic decisions in Formula 1.
This project builds a deep learning model that predicts how many laps remain until the next pit stop, using lap‑by‑lap telemetry and stint data from the 2024 season.

The goal is to support:
- race strategy simulations
- tyre degradation modelling
- stint optimisation
- proactive pit‑window identification


2. Why Pit‑Stop Prediction Matters in F1 Strategy
   
Pit stops influence:
- tyre life and degradation
- undercut/overcut opportunities
- track position
- safety car risk windows
- race pace evolution
  
Teams constantly evaluate:
- “Can we extend this stint?”
- “Will the tyres fall off?”
- “Is the undercut strong enough?”
  
A predictive model helps quantify these decisions instead of relying solely on heuristics.


3. Data Pipeline
   
Data was collected using FastF1, focusing on the 2024 season.

Data sources include:
- lap times
- compound usage
- stint lengths
- tyre age
- track status
- driver pace evolution
- session metadata

Processing steps:
- cleaning missing laps
- normalising lap times
- encoding tyre compounds
- generating rolling features
- constructing sequences for GRU input
  
The final dataset is stored in src/data/


4. Feature Engineering
5. 
Key engineered features include:
- Tyre age (laps since last pit)
- Compound type (C1–C5, inters, wets)
- Pace deltas (rolling averages, rolling std)
- Track evolution indicators
- Driver‑specific pace normalisation
- Stint progression features
These features capture tyre degradation, stint dynamics, and race evolution.


5. Model Design — GRU for Time‑Series Prediction
   
A Gated Recurrent Unit (GRU) network was chosen because:
- pit‑stop behaviour is sequential
- tyre degradation is time‑dependent
- GRUs handle long‑term dependencies efficiently
- they train faster than LSTMs with similar performance

Model Input:
Sequences of the previous N laps for a driver.

Model Output:
A single value - Predicted laps remaining until the next pit stop.


6. Training & Evaluation
   
Training setup:
- Train/validation split by race
- Sequence length: configurable
- Loss: MAE
- Optimiser: Adam
- Early stopping to prevent overfitting
  
Evaluation metrics:
- MAE (mean absolute error)
- RMSE
- Accuracy of pit‑window classification


7. Performance Analysis
   
The model successfully learns:
- tyre degradation curves
- stint length patterns
- compound‑dependent behaviour
- pace drop‑off before pit stops
  
Plots and analysis are available in the docs/ folder:
- training curves
- predicted vs actual pit‑stop laps
- tyre‑age vs pace degradation


8. Limitations
   
- Does not model safety‑car‑induced pit stops
- Driver‑specific behaviour varies significantly
- Weather‑affected races introduce noise
- Strategy decisions are multi‑factorial (traffic, gaps, risk)


9. Reproducibility
    
Install dependencies:
pip install -r requirements.txt


Run the model:
python src/train_model.py


Generate predictions:
python src/predict.py


10. Future Extensions
    
- Add safety‑car probability modelling
- Integrate tyre wear simulation
- Multi‑output model (predict stint length + tyre degradation)
- Real‑time prediction during live sessions
- Driver‑specific model fine‑tuning

11. Project Structure
    
f1-pitstop-prediction/
│
├── src/
│   ├── data/                # Processed datasets
│   ├── models/              # Saved models
│   ├── train_model.py       # Training pipeline
│   ├── predict.py           # Inference script
│
├── docs/                    # Plots, analysis, diagrams
├── requirements.txt
├── .gitignore
└── README.md


13. Acknowledgements
    
- FastF1 for data access
- Formula 1 for providing open telemetry
- GRU architecture inspired by sequence modelling research






