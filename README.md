DESCRIPTION

This project builds a data pipeline and interactive dashboard to analyze political advertising strategies in California using the Google Political Ads Transparency dataset.
The system processes raw advertising data, engineers advertiser-level behavioral features, applies multiple clustering algorithms (K-Means and Hierarchical Clustering), detects outliers using DBSCAN, and outputs a final dataset (tableau_dashboard.csv) for visualization.
An interactive Tableau dashboard is included, allowing users to explore advertiser behavior, compare clustering methods, and analyze targeting strategies, spending patterns, and efficiency metrics. The dashboard is not generated programmatically but is built on top of the pipeline output for interpretability and analysis.

INSTALLATION
Step 1 — Download Raw Data
Option A (Recommended for Reproducibility):
Download the exact dataset snapshot used for this project:
https://drive.google.com/drive/folders/1dEoctfTQRWxuXUxLEBpqajpCfpOXzpza?usp=sharing
This ensures results match those reported in the final report.

Option B (Original Source):
Go to the Google Political Ads Transparency site:
https://adstransparency.google.com/political?region=US&topic=political
Click “Export Data” → “Download CSV”
This will download a ZIP file containing multiple CSV files.
Note: Using a fresh download may produce slightly different results.

Step 2 — Download the project repository (ZIP) from GitHub
Extract the ZIP file to your desired working directory
NOTE: For grading purposes, the zip file submission contains all of this.

Step 3 — Place Raw Data in Correct Folder
Inside the project folder, locate the data/ directory
Create a subfolder named: data/raw/
Extract all CSV files from the Google dataset ZIP into this data/raw/ folder

Important:
The folder must be named exactly raw
The CSV filenames should remain unchanged

EXECUTION
Step 1 — Load Data into SQLite Database
Run the following script: 
python load_political_ads.py

What this does:
Reads raw CSV files from data/raw/
Cleans and filters relevant columns

Creates a SQLite database:
db/political_ads.db

Step 2 — Inspect Data (Optional but Recommended)
Run:
python inspect_political_ads.py
What this does:
Prints summaries of each dataset
Verifies data integrity and structure
Confirms tables were loaded correctly

Step 3 — Run Full Pipeline

Run:
python run_pipeline.py

What this does:
Performs feature engineering (California-focused advertisers)
Detects outliers using DBSCAN
Runs K-Means and Hierarchical clustering
Generates validation metrics
Outputs final dataset and visualizations

Step 4 — Locate Output Files
After running the pipeline, the following will be created:
output/
│
├── data/
│   └── tableau_dashboard.csv   ← FINAL DATASET FOR DASHBOARD
│   └── Contains other csv and json outputs such as validation_metrics.json
│
├── figures/
│   └── (clustering and validation visualizations)

Step 5 — Open Tableau Dashboard
Option A — Use Included Tableau File
Go to the Tableau/ folder
Open:
Political Ads Dashboard.twbx

If needed, reconnect the data source to:
output/data/tableau_dashboard.csv

Option B — Use Tableau Public Link
Open the .txt file in the Tableau/ folder
Copy and paste the Tableau Public link into your browser
View the fully interactive dashboard online

FINAL PROJECT STRUCTURE
After completing all steps, your directory should look like:
project_root/
│
├── data/
│   ├── raw/                ← downloaded Google data
│   └── processed/
│
├── db/
│   └── political_ads.db
│
├── output/
│   ├── data/
│   │   └── tableau_dashboard.csv
│   └── figures/
│
├── src/                   ← pipeline modules
│
├── Tableau/
│   ├── Political Ads Dashboard.twbx
│   ├── tableau_dashboard.csv
│   └── (link .txt file)
│
├── load_political_ads.py
├── inspect_political_ads.py
├── run_pipeline.py

NOTES
The dashboard uses precomputed results from the pipeline
Clustering methods (K-Means and Hierarchical) can be toggled within the dashboard
Outliers are labeled and explained directly in the dataset and UI
The Tableau dashboard is the primary interface for exploring results
