.PHONY: install extract reshape cluster shap all clean

PY ?= python3

install:
	$(PY) -m pip install -r requirements.txt

# Extract Dataset.csv from TCAD outputs (adjust --input-path to your data root)
extract:
	$(PY) Data_Extraction/main.py \
		--output-path Data_Extraction \
		--outfile Dataset.csv \
		--mostype p \
		--points 300 \
		--plt-start 1 --plt-end 3

# Reshape to long CSV and split per-sample IV files
reshape:
	$(PY) Data_Extraction/IV_ML.py \
		--input-csv Data_Extraction/Dataset.csv \
		--output-csv Data_Extraction/extracted_Id_Vgs_long.csv \
		--iv-folder Data_Extraction/IV

# Cluster IV curves and export plots/CSVs
cluster:
	$(PY) Clustering/IV_clustering.py \
		--data-path Data_Extraction/Dataset.csv \
		--output-dir Clustering/IV_Clusters \
		--points 300

# Run SHAP analysis (uses its internal config). Writes to SHAP_Analysis/SHAP_FromArrays
shap:
	cd SHAP_Analysis && $(PY) SHAP_Analysis.py

all: cluster shap

clean:
	rm -rf Clustering/IV_Clusters Data_Extraction/IV SHAP_Analysis/SHAP_FromArrays
