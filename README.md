# Lightweight Face Mask Detection from Face Images

This project is a DSCI 441 final project. It builds a lightweight classical machine-learning web app that predicts whether an uploaded face image shows a person **with a mask** or **without a mask**.

The project does **not** perform identity recognition, face recognition, person tracking, or real-time surveillance. It only performs binary image classification on face images.

## Data source

The raw image data is not included in this repository or submission zip. See `data/readme_data.txt` for download and placement instructions.

Datasets used for the final project:

1. **Face Mask Detection ~12K Images Dataset**  
   Kaggle: `ashishjangra27/face-mask-12k-images-dataset`  
   URL: `https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset`

2. **Face Mask Detection Dataset**  
   Kaggle: `omkargurav/face-mask-dataset`  
   URL: `https://www.kaggle.com/datasets/omkargurav/face-mask-dataset`

3. **Face Mask Dataset (Train-Test-Validation)**  
   Kaggle: `busrabetulcavusoglu/face-mask-dataset-tvt`  
   URL: `https://www.kaggle.com/datasets/busrabetulcavusoglu/face-mask-dataset-tvt`

After downloading, place the extracted folders inside:

```text
data/raw/
```

The code automatically searches for folders named like `WithMask`, `WithoutMask`, `with_mask`, `without_mask`, `mask`, and `no_mask`.

```text
The final data folder should look like this:
data/raw/
├── Face Mask Dataset/
│   ├── Train/
│   │   ├── WithMask/
│   │   └── WithoutMask/
│   ├── Validation/
│   │   ├── WithMask/
│   │   └── WithoutMask/
│   └── Test/
│       ├── WithMask/
│       └── WithoutMask/
│
├── extra_omkar/
│   └── Train/
│       ├── WithMask/
│       └── WithoutMask/
│
└── extra_tvt/
    ├── Train/
    │   ├── WithMask/
    │   └── WithoutMask/
    ├── Validation/
    │   ├── WithMask/
    │   └── WithoutMask/
    └── Test/
        ├── WithMask/
        └── WithoutMask/
```
## Model summary

The final model:

```text
128x128 grayscale preprocessing
+ HOG feature extraction
+ StandardScaler
+ PCA retaining 98% variance
+ RBF-kernel SVM
```

Final selected parameters:

```text
PCA n_components = 0.98
SVM C = 10.0
SVM gamma = "scale"
```

The optional tuning model `milestone_grid` is included only to reproduce the short hyperparameter search. The default model `milestone` uses the fixed final parameters so graders do not need to repeat the time-consuming grid search.

## Packages required

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main packages:

```text
streamlit
numpy
pandas
pillow
scikit-image
scikit-learn
scipy
matplotlib
joblib
kaggle
```

## How to run locally

### 1. Check the data folder

```bash
python aux_1.py --data-dir data/raw
```

### 2. Train the final fixed model

This retrains the final model using the selected parameters. It does **not** run the grid search.

```bash
python main.py train --data-dir data/raw --model-path models/mask_model.joblib --image-size 128 128 --bootstrap-samples 500
```

### 3. Evaluate a saved model

```bash
python main.py evaluate --data-dir data/raw --model-path models/mask_model.joblib --bootstrap-samples 500
```

### 4. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Then upload a face image in the browser and the app will show:

```text
Prediction: WithMask or WithoutMask
Confidence score
Class probabilities
Low-confidence warning when applicable
```

## Optional: reproduce hyperparameter tuning

The final selected parameters came from a short GridSearchCV experiment. This is optional and can take much longer than the fixed final training command.

```bash
python main.py train --data-dir data/raw --model milestone_grid --model-path models/mask_model.joblib --image-size 128 128 --bootstrap-samples 500 --n-jobs 1
```

## Repository structure

```text
folder/
│
├── ReadMe.txt
├── README.md
├── requirements.txt
├── main.py
├── aux_1.py
│
├── data/
│   └── readme_data.txt
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── models.py
│   ├── predict.py
│   └── train.py
│
└── app/
    └── streamlit_app.py
```

## Notes for deployment

The Streamlit app expects the trained model file at:

```text
mask_model.joblib
```

If deploying on Streamlit Community Cloud, make sure `requirements.txt`, `app/streamlit_app.py`, and `models/mask_model.joblib` are included in the GitHub repository. If the model file is too large for GitHub, use Git LFS or retrain a smaller final model.
