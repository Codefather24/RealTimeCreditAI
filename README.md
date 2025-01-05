# CreditPulse

An AI-powered system for real-time credit risk assessment using alternative data sources like social media sentiment, utility payments, and geolocation. This project combines machine learning, natural language processing (NLP), and explainable AI to deliver accurate and transparent credit scores.

## Features

- **Data Integration:** Combines traditional credit metrics with alternative data sources.
- **Sentiment Analysis:** Extracts financial insights using NLP from social media activity.
- **Credit Risk Model:** Predicts creditworthiness using machine learning.
- **Explainability:** Provides transparent credit risk scores to enhance trust.
- **Dashboard:** Visual interface for lenders to explore credit insights.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Codefather24/RealTimeCreditAI.git
   cd RealTimeCreditAI
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run Data Preprocessing:**
   Prepare traditional and alternative data sources.
   ```bash
   python preprocess_data.py
   ```

2. **Train the Model:**
   Train the credit risk assessment model.
   ```bash
   python train_model.py
   ```

3. **Evaluate the Model:**
   Evaluate model performance and explainability.
   ```bash
   python evaluate_model.py
   ```

4. **Run the Dashboard:**
   Launch the dashboard for visualizing credit insights.
   ```bash
   streamlit run dashboard.py
   ```

## Datasets

- [Home Credit Default Risk Dataset](https://www.kaggle.com/competitions/home-credit-default-risk)
- [Geolife GPS Trajectories](https://www.microsoft.com/en-us/download/details.aspx?id=52367)
- [Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)


## License

This project is licensed under the [MIT License](LICENSE).

## Contact

- **Email:** thecodefather24@gmail.com
- **GitHub:** [https://github.com/yourusername/CreditPulse](https://github.com/Codefather24/RealTimeCreditAI)
