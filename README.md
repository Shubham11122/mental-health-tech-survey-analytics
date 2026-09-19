# Mental Health in Tech Survey — Analytics

An end-to-end data analysis project on the **OSMI 2014 Mental Health in Tech Survey**, covering data cleaning, exploratory data analysis, and an interactive Streamlit dashboard with a logistic regression risk predictor.

## 🔗 Live Demo

**[https://mental-health-tech-survey-analytics.streamlit.app/](https://mental-health-tech-survey-analytics.streamlit.app/)**

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
├── EDA_Visuals.ipynb            # 19 exploratory visuals with findings
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

## 📈 Exploratory Data Analysis — 19 Visuals

| # | Visual Title |
|---|---|
| 1 | What is the age distribution of employees who participated in the survey? |
| 2 | What is the gender distribution of the survey respondents? |
| 3 | Which countries have the highest number of survey respondents? |
| 4 | How common is a family history of mental illness among survey respondents? |
| 5 | How many respondents have sought treatment for a mental health condition? |
| 6 | Is family history associated with seeking mental health treatment? |
| 7 | Among respondents with a mental health condition, how often does it interfere with their work? |
| 8 | How does company size relate to mental health treatment-seeking? |
| 9 | Does working remotely relate to mental health treatment-seeking? |
| 10 | Does working for a tech company relate to mental health treatment-seeking? |
| 11 | How widely do employers provide mental health benefits? |
| 12 | Are employees aware of the mental health care options provided by their employers? |
| 13 | How easy is it for employees to take medical leave for a mental health condition? |
| 14 | Do employees perceive negative consequences from discussing mental health with their employer? |
| 15 | How willing are employees to discuss mental health with coworkers and supervisors? |
| 16 | Is observing negative consequences for coworkers associated with willingness to discuss mental health? |
| 17 | How does treatment-seeking vary across different age groups? |
| 18 | How do age, treatment-seeking, and work interference vary together? |
| 19 | How do treatment-seeking, remote work, and age vary across major respondent countries? |

## 🖥️ Dashboard Features (`app.py`)

- **Overview** — demographics and treatment-seeking summary
- **Workplace Factors** — every major policy variable compared against treatment-seeking
- **Correlation Explorer** — heatmap and pair plot of key encoded variables
- **Risk Predictor** — a logistic regression trained live on the data; adjust sliders/dropdowns to get a live predicted likelihood of seeking treatment
- **Data Explorer** — searchable, filterable table with CSV export
- **Insights** — key findings and recommendations

All views respond to sidebar filters: country, gender, age range, company size, and tech-company-only.

## 🔍 Findings

- **Family history is the strongest single predictor of treatment-seeking.** Respondents with a family history of mental illness sought treatment at roughly **74%**, compared to about **36%** among those without one.
- **Work interference tracks closely with treatment-seeking.** Respondents who report their condition interferes with work "Often" seek treatment at far higher rates than those who report "Never" or "Not Applicable."
- **The respondent base skews young and male** — concentrated in the 25–40 age range, with 78.8% Male, 19.7% Female, and 1.5% Other. Conclusions about smaller demographic slices carry more uncertainty.
- **Treatment-seeking is close to an even split overall** (50.6% Yes / 49.4% No), showing it is a common experience rather than a rare event in this population.
- **Awareness gaps are as large as policy gaps.** For benefits, care options, and leave difficulty, a large share of respondents answered "Don't know" — behaving statistically much closer to "No" than to "Yes."
  - Only 38% of respondents know their employer provides mental health benefits; 33% don't know.
  - Among respondents *without* employer benefits, only 38.6% know about available care options — versus 61.4% among those *with* benefits.
  - Over 550 respondents (the largest single group) don't know how easy it is to take medical leave for a mental health condition.
- **Company size and remote-work status show weak, inconsistent relationships** with treatment-seeking — neither is a reliable standalone lever.
- **Treatment-seeking generally rises with age** through the younger-to-middle age groups, from about 45% (18–24) to over 55% (40–44), before leveling off.
- **The United States dominates the sample** (751 of 1,255 respondents), so findings are most representative of a US tech-workplace context.

## 💡 Business Recommendations

1. **Communicate existing benefits, care options, and leave policy clearly and repeatedly.** The gap between "Yes" and "Don't know" responses is often as large as the gap between "Yes" and "No" — meaning many employees already have support they're unaware of. This is a near-zero-cost fix.
2. **Train managers to recognize early signs of work interference.** Treatment-seeking rises sharply once interference is reported as frequent, so earlier recognition could shorten time-to-treatment.
3. **Use voluntary, anonymous outreach informed by family-history risk signals**, without requiring individual disclosure — e.g., periodic, opt-in mental health check-ins available to everyone.
4. **Extend support to smaller companies and self-employed workers**, who are more likely to fall outside standard employer benefit structures.
5. **State anonymity guarantees explicitly** wherever mental health resources are described, since uncertainty about anonymity is common and likely discourages treatment-seeking.
6. **Avoid relying on demographic targeting** (age, gender, remote-work status, company size) as a primary lever — the workplace policy factors above show a stronger, more actionable relationship with treatment-seeking.

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

## 👤 Author

**Shubham** — [GitHub](https://github.com/Shubham11122)

## 📄 License

This project is for educational purposes, using publicly available survey data from Kaggle.
