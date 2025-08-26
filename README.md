## Model Inference Dashboard

An interactive Streamlit dashboard to explore model inference results, sourced from `./data/experiments.csv`.

### Features
- Filters for date and categorical columns
- KPI cards (rows, mean, median, std for a selected metric)
- Charts: time series by date, bar by category
- Data table with the filtered view
- Export filtered data to CSV/Excel and a Markdown summary report

### Requirements
Python 3.9+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run

Ensure your data file exists at `./data/experiments.csv` relative to the project root (`/Users/yuval/repository/leader-board/data/experiments.csv`). Then run:

```bash
streamlit run streamlit_app.py
```

The app will open in your browser. Use the sidebar to set filters and metrics. Use the Export section to download CSV, Excel, or a Markdown report.


