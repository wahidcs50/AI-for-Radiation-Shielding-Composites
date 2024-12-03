
from Modeling.ml import DataHandler, ModelTrainer, ModelEvaluator, Visualization
from FNRCS_Data.data_cleaning import load_and_clean_data, parse_composition
import os
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

def main():
    try:
        process_and_save_data(DATA_PATH, DIRECTORY_PATH, FILE_NAME)
        ml_training_plotting()
    except Exception as e:
        print(f"An unexpected error occurred in the main function: {e}")

if __name__ == "__main__":
    main()
