Lightweight Face Mask Detection from Face Images

This project predicts whether an uploaded face image is WithMask or WithoutMask.
It uses a classical, non-deep-learning pipeline:

  128x128 grayscale preprocessing
  + HOG feature extraction
  + StandardScaler
  + PCA retaining 98% variance
  + RBF SVM with C=10.0 and gamma="scale"

Data:
Raw datasets are not included. See data/readme_data.txt for Kaggle download
links and placement instructions.

Install:
  pip install -r requirements.txt

Check data:
  python aux_1.py --data-dir data/raw

Train final fixed model, without repeating grid search:
  python main.py train --data-dir data/raw --model-path models/mask_model.joblib --image-size 128 128 --bootstrap-samples 500

Evaluate saved model:
  python main.py evaluate --data-dir data/raw --model-path models/mask_model.joblib --bootstrap-samples 500

Run Streamlit demo:
  streamlit run app/streamlit_app.py

Optional hyperparameter tuning, not needed for normal grading:
  python main.py train --data-dir data/raw --model milestone_grid --model-path models/mask_model.joblib --image-size 128 128 --bootstrap-samples 500 --n-jobs 1
