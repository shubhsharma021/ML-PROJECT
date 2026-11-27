# Clustering Indian States by School Infrastructure (UDISE 2022–23)

An undergraduate Machine Learning project that groups Indian States and Union Territories according to school infrastructure coverage using **K-Means clustering**.

## Research question
Can Indian States/UTs be grouped into meaningful infrastructure profiles using the availability of functional toilets, electricity, computers for pedagogical purposes, and internet connectivity?

## Dataset
- File: `data/UDISE_2022_23_Table_2.5_0.csv`
- Academic year: 2022–23 (as indicated by the supplied file name)
- Original rows: 37, including the all-India aggregate.
- Analysis observations: 36 State/UT rows; the India aggregate is excluded to avoid mixing an aggregate with individual regions.
- Original columns: 25.
- Missing values in the supplied CSV: 0.

The project uses the supplied CSV. The exact original portal URL and publication metadata were not embedded in the uploaded file, so they should be confirmed from the professor-provided source link before final submission.

## Features and preprocessing
Raw facility counts are converted to percentages of each State/UT's total schools:

1. **Toilet coverage:** mean of functional girls' toilet percentage and functional boys' toilet percentage.
2. **Electricity:** functional electricity percentage.
3. **Computers:** functional computer facility for pedagogical purposes percentage.
4. **Internet:** internet facility percentage.

The four features are standardised using `StandardScaler` before distance-based clustering. This prevents percentage dimensions with different distributions from dominating Euclidean distance. The four dimensions are equally represented; the composite index in the results is descriptive, not an official government index.

## Method
- Primary algorithm: K-Means (`random_state=42`, `n_init=50`).
- Candidate cluster counts: k = 2 to 6.
- Selection criterion: maximum silhouette score.
- Validation/interpretation: inertia (elbow), silhouette score, Davies–Bouldin index, Calinski–Harabasz score.
- Comparison: Agglomerative clustering with Ward linkage.
- Exploratory method: DBSCAN (reported separately; not treated as the primary result).
- PCA is used only to visualise the four-dimensional data in two dimensions; it is not used to create the clusters.

## Main result from the supplied data
The executed analysis selected **k = 3** using the maximum silhouette score (0.451). The resulting cluster sizes are:

- Cluster 1: 5 States/UTs
- Cluster 2: 16 States/UTs
- Cluster 3: 15 States/UTs

See `results/cluster_profile.csv` for average facility coverage by cluster and `results/state_cluster_assignments.csv` for every State/UT's cluster and feature values.

## Run locally
Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
python src/run_analysis.py
```

The script reads the CSV from `data/`, then regenerates tables in `results/` and figures in `figures/`.

## Repository structure
```text
school_infrastructure_clustering/
├── data/
│   └── UDISE_2022_23_Table_2.5_0.csv
├── src/
│   └── run_analysis.py
├── results/
│   ├── algorithm_comparison.csv
│   ├── cluster_profile.csv
│   ├── dbscan_exploration.csv
│   ├── k_selection_metrics.csv
│   └── state_cluster_assignments.csv
├── figures/
│   ├── elbow_curve.png
│   ├── silhouette_scores.png
│   ├── pca_cluster_plot.png
│   ├── cluster_profile_heatmap.png
│   └── state_infrastructure_index.png
├── report/
│   └── Research_Paper.pdf
├── README.md
└── requirements.txt
```

## Limitations
- The analysis covers one academic year only; it does not describe trends over time.
- State/UT-level averages conceal district-level and school-level inequalities.
- A high percentage of facilities does not by itself establish quality, usability, accessibility, or uninterrupted service.
- The four dimensions are equally weighted for the descriptive index; alternative weights may change rankings.
- Clustering is exploratory and does not establish causal relationships or prove that one state’s education system is better overall.

## Academic integrity and reproducibility
All reported numerical findings should be regenerated from the included CSV and Python script. Review the outputs, understand the feature construction and clustering steps, and personalise the author/supervisor details in the paper before submission. Disclose AI assistance according to your college's rules.

## Author details to complete
- Student name: [Your Name]
- Roll number: [Your Roll Number]
- College: [Your College]
- Supervisor: [Professor's Name]
