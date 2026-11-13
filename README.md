# Shipment ETA & Delay Intelligence

A compact end-to-end ML assignment project that predicts final shipment delay at the **port-arrival checkpoint**, models delay propagation, estimates the probability of a delay greater than 24 hours, and supports what-if scenario analysis.

## 1. Problem definition

- Scheduled ETA: planned final delivery timestamp.
- Actual delay: `actual_delivery_timestamp - scheduled_delivery_timestamp`, in hours.
- Prediction checkpoint: immediately after port arrival.
- Regression target: final delivery delay in hours.
- Classification target: whether final delivery is more than 24 hours late.
- Predicted ETA: scheduled delivery timestamp + predicted final delay.

  A shipment may experience:
- Departure delays due to operational constraints.
- Transit delays caused by weather conditions.
- Port congestion and vessel scheduling issues.
- Customs clearance delays.
- Additional delays during the final delivery stage.

The model uses only information available at the port-arrival checkpoint: route distance,  departure/transit delays,  port-arrival delay,  congestion,  weather severity,  customs complexity,  planned customs duration,  schedule slack and transshipment count. Future actual customs and delivery timestamps are excluded from the feature set.

The Solution #1
An AI-powered shipment intelligence system that performs four primary tasks:
1. ETA Prediction: Predicts the expected final delivery delay in hours and calculates the estimated delivery timestamp.
2. Delay Risk Classification: Calculates the probability that a shipment will be delayed by more than 24 hours.
3. Delay Propagation Analysis: Explains how upstream delays carry forward through subsequent shipment stages.
4. Explainable AI: Identifies the operational variables influencing the prediction and supports what-if scenario analysis.
The application is developed using Python, Scikit-learn, Pandas, NumPy and Streamlit.

## 2. Delay propagation

The transparent stage transition is:

`carried_delay = max(0, inherited_delay - schedule_slack - recovery)`

`resulting_delay = carried_delay + newly_introduced_delay`

The ML model complements this rule-based mechanism by learning nonlinear relationships between operational conditions and final delay. The dashboard displays both an ML prediction and a transparent rule baseline.

## 3. Dataset

The generator creates 300 reproducible synthetic journeys (seed 42) with scheduled and actual timestamps for departure,  port arrival,   customs and final delivery.  Congestion,  weather,  customs complexity,  transshipments,  schedule slack,  recovery and stochastic disruptions affect the simulated outcomes.

Synthetic data is useful for demonstrating methodology and testing code, but cannot establish real-world predictive performance.

#Synthetic Data Generation Details-
Dataset characteristics-

Parameter			                Implementation
Total shipment records		    300
Random seed			              42
Route origins			            5 sampled locations
Route destinations		        5 sampled locations
Route distance		          	2,500–14,500 km
Planned speed			            25–36 km/h
Congestion index	           	0–100
Weather severity		          0–10
Customs complexity		        1–10
Transshipments		          	0–3
Schedule slack		          	0–18 hours
Delay classification threshold  24 hours

The synthetic schedule dates are distributed over a 180-day window beginning on 1 January 2025.

Dataset outcome
The generated dataset contains 300 shipment records and 26 columns.

The dataset's final delay distribution has:
- Mean delay: approximately 12.28 hours.
- Standard deviation: approximately 8.50 hours.
- Shipments exceeding 24 hours of delay: 30.
- Severe-delay proportion: 10%.

Feature selection
Selected ten input features:
Feature                      	Purpose
distance_km			              Represents route distance
departure_delay_hours		      Captures departure disruption
transit_delay_hours		        Captures upstream transit delay
port_arrival_delay_hours	    Represents accumulated delay at port arrival
port_congestion_index		      Measures port congestion
weather_severity		          Represents environmental disruption
customs_complexity		        Estimates customs-processing difficulty
planned_customs_hours		      Represents planned clearance duration
schedule_slack_hours		      Represents available schedule buffer
transshipment_count		        Captures additional transfer complexity

These features are used by both machine-learning models.

Feature	Contribution direction
High port congestion			    Increases predicted delay
High customs complexity		   	Increases predicted delay
High schedule slack		      	Reduces predicted delay
Severe weather				        May increase predicted delay
Additional transshipments		  May increase predicted delay

## 4. Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python src/generate_data.py
python src/train.py
streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

## 5. Run tests

```bash
pytest -q
```

## 6. Repository map


shipment_eta_intelligence/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── generate_data.py
│   ├── train.py
│   └── propagation.py
│
├── data/
│   └── synthetic_shipments.csv
│
├── models/
│   ├── eta_regressor.joblib
│   ├── delay_classifier.joblib
│   └── model_metrics.json
│
├── reports/
│   └── technical_report.pdf
│
├── tests/
│   └── test_propagation.py
│
├── requirements.txt
├── README.md
└── .gitignore
└── flowchart.png


- `src/generate_data.py`: synthetic shipment generator.
- `src/propagation.py`: transparent stage-delay calculations and baseline.
- `src/train.py`: regression/classification training and evaluation.
- `app/streamlit_app.py`: interactive dashboard and scenario simulator.
- `data/synthetic_shipments.csv`: generated 300-record dataset.
- `models/`: saved models and evaluation metrics.
- `reports/technical_report.pdf`: concise technical report.
- `tests/`: propagation unit tests.

## 7. Evaluation

Regression metrics: MAE, RMSE and R². Classification metrics: precision, recall, F1, ROC-AUC and Brier score. A fixed 80/20 train/test split and seed 42 are used for reproducibility. Model comparison is reported on the held-out test split; for a larger production study, select hyperparameters using a separate validation set and preserve a final untouched test set.

## 8. Limitations

- All data is synthetic; external validity is not established.
- Only one prediction checkpoint is implemented (port arrival).
- The 24-hour classification threshold is a project assumption.
- SHAP is used for local model attribution when installed; attribution is not causal proof.
- Scenario outputs are model-based estimates, not guaranteed operational savings.
- The compact synthetic dataset is intended for an assignment demonstration, not deployment.
