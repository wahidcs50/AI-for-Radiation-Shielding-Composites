
from Modeling.ml import DataHandler, ModelTrainer, ModelEvaluator, Visualization
from Modeling.ann import Model, Trainer, Visualizer, EarlyStopping
from FNRCS_Data.data_cleaning import load_and_clean_data, parse_composition
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import torch
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
DIRECTORY_PATH = os.getenv("SAVE_PATH")
FILE_NAME = os.getenv("FILE_NAME")
file_path = os.path.join(DIRECTORY_PATH, FILE_NAME)

def process_and_save_data(data_path: str, save_path: str, file_name: str):
    try:
        clean_data, total_sheets = load_and_clean_data(data_path)
        clean_data = clean_data.replace('', pd.NA)
        clean_data = clean_data.infer_objects()
        combined_data = clean_data.dropna(axis=1, how='all')
        parsed_data = combined_data['composition'].apply(parse_composition)
        parsed_df = pd.DataFrame(list(parsed_data))

        for comp in parsed_df.columns:
            combined_data[f'{comp}_fraction'] = parsed_df[comp]
        
        print("Combined Data Preview:")
        print(combined_data.head())
        output_path = os.path.join(save_path, file_name)
        combined_data.to_excel(output_path, index=False)
        print(f"Data successfully saved to {output_path}")

    except Exception as e:
        print(f"Error occurred while processing data: {e}")

def ml_training_plotting():
    try:
        
        if not DIRECTORY_PATH or not FILE_NAME:
            raise ValueError("Environment variables SAVE_PATH and FILE_NAME must be set.")
        print(f"Loading data from: {file_path}")

        data_handler = DataHandler(file_path)
        print("DataHandler initialized.")

        df = data_handler.load_data()
        if df.empty:
            raise ValueError("The loaded dataset is empty.")
        print("Data Loaded:")
        print(df.head())

    except Exception as e:
        print(f"Error loading or initializing data: {e}")
        return

    try:
        X_train, X_test, y_train, y_test = data_handler.preprocess_data()
        trainer = ModelTrainer(X_train, y_train, X_test, y_test)
        xgb_model = trainer.train_xgboost()
        rf_model = trainer.train_random_forest()

    except Exception as e:
        print(f"Error during model training: {e}")
        return

    try:
        evaluator_xgb = ModelEvaluator(xgb_model, X_test, y_test)
        y_pred_xgb = evaluator_xgb.evaluate()
        print('XGBoost predictions:', y_pred_xgb)
        evaluator_xgb.plot_shap_summary()
        evaluator_rf = ModelEvaluator(rf_model, X_test, y_test)
        y_pred_rf = evaluator_rf.evaluate()
        print('Random Forest predictions:', y_pred_rf)
        evaluator_rf.plot_shap_summary()

    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return

    try:
        X = df[['Density:', 'AMW:', 'C2H6OSi_fraction', 'CdO_fraction', 'B4C_fraction', 'Gd2O3_fraction']]
        X_test_df = pd.DataFrame(X_test, columns=X.columns)

        Visualization.plot_actual_vs_predicted(y_test, y_pred_xgb, 'XGBoost FNRCS Prediction vs Actual')
        Visualization.plot_actual_vs_predicted(y_test, y_pred_rf, 'Random Forest FNRCS Prediction vs Actual')

        for feature in ['C2H6OSi_fraction', 'CdO_fraction', 'B4C_fraction', 'Gd2O3_fraction']:
            x = X_test_df['Density:']
            y = X_test_df[feature]
            z = y_pred_rf
            Visualization.graph_ploting(x, y, z)
        print("Visualization completed successfully.")

    except Exception as e:
        print(f"Error during visualization: {e}")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
torch.manual_seed(0)
np.random.seed(0)
def ann_training_ploting():
    try: 
        if not DIRECTORY_PATH or not FILE_NAME:
            raise ValueError("Environment variables SAVE_PATH and FILE_NAME must be set.")
        print(f"Loading data from: {file_path}")

        data_handler = DataHandler(file_path)
        print("DataHandler initialized.")

        df = data_handler.load_data()
        if df.empty:
            raise ValueError("The loaded dataset is empty.")
        print("Data Loaded:")
        print(df.head())
        X_train, X_test, y_train, y_test = data_handler.preprocess_data()
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, worker_init_fn=seed_worker, generator=g)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, worker_init_fn=seed_worker, generator=g)

        model = Model(input_size=X_train.shape[1])

        trainer = Trainer(model, train_loader, test_loader)
        early_stopping = EarlyStopping(patience=20, min_delta=0.001) 
        train_losses, val_losses = trainer.train(epochs=500, early_stopping=early_stopping)
        y_test_actual = y_test_tensor.cpu().numpy()
        all_predictions= trainer.evaluate()
        print("All predictions", all_predictions)
        scores= trainer.scores(y_test_actual, all_predictions)

        Visualizer.plot_loss(train_losses, val_losses)
        Visualizer.plot_predictions(y_test_actual, all_predictions, train_losses, val_losses)
    except Exception as e:
        print(f'Error while trianing ann :', {e})
        return

def main():
    try:
        process_and_save_data(DATA_PATH, DIRECTORY_PATH, FILE_NAME)
        ml_training_plotting()
        ann_training_ploting()
    except Exception as e:
        print(f"An unexpected error occurred in the main function: {e}")

if __name__ == "__main__":
    main()
