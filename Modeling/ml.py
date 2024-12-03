import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import shap
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
    def load_data(self):
        self.df = pd.read_excel(self.file_path)
        return self.df
    def preprocess_data(self):
        X = self.df[['Density:', 'AMW:', 'C2H6OSi_fraction', 'CdO_fraction', 'B4C_fraction', 'Gd2O3_fraction']]
        y = self.df['FNRCS:']
        return train_test_split(X, y, test_size=0.2, random_state=42)
class ModelTrainer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
    def train_xgboost(self):
        model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3,
                                 learning_rate=0.1, max_depth=5, alpha=10, n_estimators=100)
        model.fit(self.X_train, self.y_train)
        return model
    def train_random_forest(self):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        return model
class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        print(f'R² Score: {r2:.4f}')
        print(f'MSE: {mse:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'MAE: {mae:.4f}')
        return y_pred
    def plot_shap_summary(self):
        explainer = shap.Explainer(self.model, self.X_test)
        shap_values = explainer(self.X_test)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, self.X_test, show=False)
        # plt.savefig('shap_summary.png', dpi=300)
        plt.show()
class Visualization:
    @staticmethod
    def plot_actual_vs_predicted(y_test, y_pred, title):
        plt.figure(figsize=(6,4))
        sns.regplot(x=y_test, y=y_pred, scatter_kws={'s':10}, line_kws={'color':'red'})
        plt.xlabel('Actual Values', fontweight='bold')
        plt.ylabel('Predicted Values', fontweight='bold')
        plt.title(title, fontweight='bold')
        plt.legend(['Prediction','Best Fit'], loc='best', prop={'size': 10, 'weight': 'bold'})
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')  
        # plt.savefig('xglinear_fit_plot.png', dpi=600)
        plt.show()
    def graph_ploting(x,y,z):  
        xyz = np.vstack([x, y, z])
        density = gaussian_kde(xyz)(xyz)
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        scat = ax.scatter(x, y, z, c=density, cmap='plasma')  
        
        cbar = fig.colorbar(scat, ax=ax, orientation='vertical', pad=0.06, anchor=(0, 0.45), shrink=0.5)
        cbar.set_label('Density', fontsize=12, fontweight='bold') 
        cbar.ax.tick_params(labelsize=10, width=2, colors='black') 
        
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight('bold') 
        ax.set_xlabel('Composit Density', fontweight='bold')
        ax.set_ylabel('C2H6OSi Fraction', fontweight='bold')
        ax.set_zlabel('FNRCS', fontweight='bold')
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        for label in ax.get_zticklabels():
            label.set_fontweight('bold')
        
        # plt.savefig('denisty1.png', dpi=600)
        plt.show()

if __name__ == "__main__":
    data_handler = DataHandler('/kaggle/input/second-clean-data.xlsx')
    df = data_handler.load_data()
    X_train, X_test, y_train, y_test = data_handler.preprocess_data()

    trainer = ModelTrainer(X_train, y_train, X_test, y_test)

    xgb_model = trainer.train_xgboost()
    rf_model = trainer.train_random_forest()
    
    evaluator_xgb = ModelEvaluator(xgb_model, X_test, y_test)
    y_pred_xgb = evaluator_xgb.evaluate()
    evaluator_xgb.plot_shap_summary()

    evaluator_rf = ModelEvaluator(rf_model, X_test, y_test)
    y_pred_rf = evaluator_rf.evaluate()
    evaluator_rf.plot_shap_summary()

    X=df[['Density:', 'AMW:', 'C2H6OSi_fraction', 'CdO_fraction', 'B4C_fraction', 'Gd2O3_fraction']]
    X_test_df = pd.DataFrame(X_test, columns=X.columns)
    Visualization.plot_actual_vs_predicted(y_test, y_pred_xgb, 'XGBoost FNRCS Prediction vs Actual')
    Visualization.plot_actual_vs_predicted(y_test, y_pred_rf, 'Random Forest FNRCS Prediction vs Actual')
    for i in X[['C2H6OSi_fraction','CdO_fraction','B4C_fraction','Gd2O3_fraction']]:
        x = X_test_df['Density:']
        y = X_test_df[i]
        z = y_pred_rf
        Visualization.graph_ploting(x,y,z)
    
