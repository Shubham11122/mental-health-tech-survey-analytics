# Mental Health in Tech Survey — Analytics

An end-to-end data analysis project on the **OSMI 2014 Mental Health in Tech Survey**, covering data cleaning, exploratory data analysis, and an interactive Streamlit dashboard with a logistic regression risk predictor.

## 🎯 Problem Statement

Mental health remains under-discussed in tech workplaces, and many companies don't know whether their policies (benefits, anonymity protections, leave policy, manager/coworker support) actually correlate with employees seeking treatment. This project analyzes the 2014 OSMI survey to identify which workplace and demographic factors are most associated with an employee seeking treatment for a mental health condition.

## 🎯 Business Objective

Identify the factors most strongly associated with treatment-seeking, so an organization can prioritize the policies most likely to increase help-seeking and reduce the stigma tied to disclosing a mental health condition at work.

## 📊 Dataset

- **Source:** [OSMI Mental Health in Tech Survey, 2014 (Kaggle)](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
- **Raw size:** 1,259 responses × 27 columns
- **Cleaned size:** 1,255 responses × 26 columns

## 🗂️ Project Structure

```
├── data_cleaning.ipynb          # Cleans raw survey data
├── EDA_Visuals.ipynb            # 19+ exploratory visuals with findings
├── app.py                       # Streamlit dashboard
├── cleaned_survey_readable.csv  # Cleaned, analysis-ready dataset
├── requirements.txt
├── .streamlit/
│   └── config.toml              # App theme
└── README.md
```

## 🧹 Data Cleaning

- Fixed `Age` outliers (negative values, and one entry of `99999999999`) by clipping to a plausible 18–75 range and imputing with the median.
- Normalized `Gender` from 49 raw free-text spellings into consistent categories (Male / Female / Other).
- Filled missing `state`, `self_employed`, and `work_interfere` with meaningful defaults rather than dropping rows.
- Removed duplicate responses (detected only after normalizing `Gender`, and excluding the always-unique `Timestamp` column from the check).

## 📈 Exploratory Data Analysis

19+ visuals covering:
- Respondent demographics (age, gender, country)
- Treatment-seeking overview
- Family history, work interference, and company-size effects
- Employer benefits, care options, wellness programs, and leave policy
- Coworker/supervisor comfort and observed workplace consequences

## 🖥️ Dashboard Features (`app.py`)

- **Overview** — demographics and treatment-seeking summary
- **Workplace Factors** — every major policy variable compared against treatment-seeking
- **Correlation Explorer** — heatmap and pair plot of key encoded variables
- **Risk Predictor** — a logistic regression trained live on the data; adjust sliders/dropdowns to get a live predicted likelihood of seeking treatment
- **Data Explorer** — searchable, filterable table with CSV export
- **Insights** — key findings and recommendations

All views respond to sidebar filters: country, gender, age range, company size, and tech-company-only.

## 🚀 Run Locally

```bash
git clone https://github.com/Shubham11122/mental-health-tech-survey-analytics.git
cd mental-health-tech-survey-analytics
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select this repo, branch `main`, main file `app.py`.
4. Click **Deploy**.

## 🛠️ Tech Stack

- **Python**, **Pandas**, **NumPy** — data cleaning & analysis
- **Matplotlib**, **Seaborn** — static EDA visuals
- **Plotly** — interactive dashboard charts
- **Scikit-learn** — logistic regression risk model
- **Streamlit** — web application framework

## 📌 Key Insights

- Family history and self-reported work interference are the strongest individual correlates of treatment-seeking.
- Awareness gaps matter as much as policy gaps — "don't know" responses for benefits, care options, and leave behave much like "no."
- Remote work, company size, age, and gender show weak standalone relationships with treatment-seeking.

## 👤 Author

**Shubham** — [GitHub](https://github.com/Shubham11122)

## 📄 License

This project is for educational purposes, using publicly available survey data from Kaggle.
