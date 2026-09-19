# Mental Health in Tech Survey — Streamlit Dashboard

Interactive analysis of the OSMI 2014 Mental Health in Tech Survey:
demographics, workplace factors, correlation explorer, a logistic
regression risk predictor, a filterable data explorer, and insights.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

If colors look off (dark theme showing instead of the intended light
one), force the theme explicitly:

```bash
streamlit run app.py --theme.base light --theme.primaryColor "#2F6F6B" --theme.backgroundColor "#FFFFFF" --theme.secondaryBackgroundColor "#F4F5F6" --theme.textColor "#1F2328"
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository (see steps below).
2. Go to https://share.streamlit.io → "New app".
3. Pick this repo, branch `main`, main file `app.py`.
4. Click Deploy.

Data: OSMI Mental Health in Tech Survey, 2014 (Kaggle).
