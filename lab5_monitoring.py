import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json
from datetime import datetime
import warnings
import sys
warnings.filterwarnings('ignore')

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, ClassificationPreset, DataQualityPreset
    EVIDENTLY_AVAILABLE = True
except ImportError as e:
    print(f"Помилка імпорту Evidently: {e}")
    EVIDENTLY_AVAILABLE = False
    sys.exit(1)

import os

DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'breathwell_clinic'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
    'port': os.getenv('DB_PORT', '5432')
}

def safe_print(message):
    try:
        print(message)
    except UnicodeEncodeError:
        safe_message = message.replace('✓', '[OK]').replace('✗', '[ERROR]').replace('✅', '[SUCCESS]').replace('❌', '[FAILED]')
        print(safe_message)

def get_engine():
    connection_string = (
        f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}"
        f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}"
        f"/{DATABASE_CONFIG['database']}"
    )
    return create_engine(connection_string)

def load_reference_data(engine):
    safe_print("Завантаження еталонних даних")
    
    try:
        query = """
        SELECT 
            patientid, timestamp,
            age, gender, bmi, smoking, physicalactivity, dietquality, sleepquality,
            pollutionexposure, pollenexposure, dustexposure,
            petallergy, familyhistoryasthma, historyofallergies, eczema, hayfever,
            gastroesophagealreflux, lungfunctionfev1, lungfunctionfvc,
            wheezing, shortnessofbreath, chesttightness, coughing, nighttimesymptoms,
            exerciseinduced,
            ethnicity_1, ethnicity_2, ethnicity_3,
            educationlevel_1, educationlevel_2, educationlevel_3,
            diagnosis as target
        FROM asthma_patients_data
        ORDER BY timestamp
        """
        
        df = pd.read_sql(query, engine)
        df['prediction'] = df['target']
        
        safe_print(f"Завантажено {len(df)} еталонних записів")
        return df
        
    except Exception as e:
        safe_print(f"Помилка завантаження еталонних даних: {e}")
        return pd.DataFrame()

def load_current_data(engine):
    safe_print("Завантаження поточних даних")
    
    try:
        query = """
        SELECT 
            p.patientid, p.created_at as timestamp,
            p.predicted_label as prediction, p.predicted_proba,
            p.true_label as target, i.input_data
        FROM predictions p
        LEFT JOIN inference_inputs i ON p.patientid = i.patientid
        WHERE p.source = 'inference'
        ORDER BY p.created_at
        """
        
        df = pd.read_sql(query, engine)
        
        if len(df) == 0:
            safe_print("Немає даних інференсу, генерація синтетичних даних")
            return generate_synthetic_data(engine)
        
        if 'input_data' in df.columns:
            input_data_expanded = df['input_data'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else (x if x is not None else {})
            )
            input_features_df = pd.DataFrame(input_data_expanded.tolist())
            df = pd.concat([df.drop('input_data', axis=1), input_features_df], axis=1)
        
        safe_print(f"Завантажено {len(df)} поточних записів")
        return df
        
    except Exception as e:
        safe_print(f"Помилка завантаження поточних даних: {e}")
        return pd.DataFrame()

def generate_synthetic_data(engine):
    safe_print("Генерація синтетичних даних")
    
    try:
        query = "SELECT * FROM asthma_patients_data ORDER BY RANDOM() LIMIT 100"
        df = pd.read_sql(query, engine)
        
        numeric_cols = ['age', 'bmi', 'pollutionexposure', 'pollenexposure', 'dustexposure', 
                        'lungfunctionfev1', 'lungfunctionfvc']
        
        for col in numeric_cols:
            if col in df.columns:
                noise = np.random.normal(0.15, 0.25, len(df))
                df[col] = df[col] + noise
        
        df['timestamp'] = pd.date_range(end=datetime.now(), periods=len(df), freq='1H')
        df = df.rename(columns={'diagnosis': 'target'})
        df['prediction'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
        
        safe_print(f"Згенеровано {len(df)} синтетичних записів")
        return df
    except Exception as e:
        safe_print(f"Помилка генерації синтетичних даних: {e}")
        return pd.DataFrame()

def prepare_data(reference_df, current_df):
    safe_print("Підготовка даних")
    
    exclude_cols = ['patientid', 'timestamp', 'target', 'prediction', 'predicted_proba', 'doctorincharge']
    common_features = [col for col in reference_df.columns 
                      if col in current_df.columns and col not in exclude_cols]
    
    ref_final = reference_df[common_features + ['timestamp', 'target', 'prediction']].copy()
    curr_final = current_df[common_features + ['timestamp', 'target', 'prediction']].copy()
    
    for col in common_features:
        ref_final[col] = pd.to_numeric(ref_final[col], errors='coerce')
        curr_final[col] = pd.to_numeric(curr_final[col], errors='coerce')
    
    ref_final = ref_final.fillna(0)
    curr_final = curr_final.fillna(0)
    
    safe_print(f"Еталонні: {ref_final.shape}, Поточні: {curr_final.shape}")
    return ref_final, curr_final

def create_column_mapping():
    numerical_features = [
        'age', 'bmi', 'pollutionexposure', 'pollenexposure', 'dustexposure',
        'lungfunctionfev1', 'lungfunctionfvc', 'physicalactivity', 'dietquality',
        'sleepquality'
    ]
    
    categorical_features = [
        'gender', 'smoking', 'petallergy', 'familyhistoryasthma', 
        'historyofallergies', 'eczema', 'hayfever', 'gastroesophagealreflux',
        'wheezing', 'shortnessofbreath', 'chesttightness', 'coughing',
        'nighttimesymptoms', 'exerciseinduced',
        'ethnicity_1', 'ethnicity_2', 'ethnicity_3',
        'educationlevel_1', 'educationlevel_2', 'educationlevel_3'
    ]
    
    column_mapping = ColumnMapping()
    column_mapping.target = 'target'
    column_mapping.prediction = 'prediction'
    column_mapping.datetime = 'timestamp'
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features
    
    return column_mapping

def generate_combined_dashboard(reference_df, current_df, column_mapping):
    safe_print("Генерація дашборду")
    
    try:
        report = Report(metrics=[
            DataQualityPreset(),
            DataDriftPreset(),
            TargetDriftPreset(),
            ClassificationPreset(),
        ])
        
        report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=column_mapping
        )
        
        html_content = report.get_html()
        
        custom_html = f"""
<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Моніторинг ML Моделі - Лабораторна 5</title>
    <style>
        body {{ font-family: Arial; background: #f5f5f5; margin: 0; padding: 20px; }}
        .header {{ background: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; text-align: center; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Моніторинг ML Моделі</h1>
        <h2>Лабораторна робота 5: Evidently AI Дашборд</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div style="font-size: 2rem; font-weight: bold;">{len(reference_df)}</div>
                <div>Еталонні дані</div>
            </div>
            <div class="stat-card">
                <div style="font-size: 2rem; font-weight: bold;">{len(current_df)}</div>
                <div>Поточні дані</div>
            </div>
            <div class="stat-card">
                <div style="font-size: 2rem; font-weight: bold;">{len(reference_df.columns) - 3}</div>
                <div>Ознаки</div>
            </div>
        </div>
    </div>
    
    <div style="background: white; padding: 30px; border-radius: 10px;">
        <h2>Звіти моніторингу Evidently AI</h2>
        {html_content}
    </div>
    
    <div style="text-align: center; margin-top: 30px; color: #666;">
        <p><strong>Asthma Prediction API</strong> - Система моніторингу моделі діагностики астми</p>
        <p>Згенеровано автоматично: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</p>
    </div>
</body>
</html>
"""
        
        with open('lab5_monitoring_dashboard.html', 'w', encoding='utf-8') as f:
            f.write(custom_html)
        
        safe_print("Дашборд успішно збережено: lab5_monitoring_dashboard.html")
        return True
        
    except Exception as e:
        safe_print(f"Помилка генерації дашборду: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    safe_print("=" * 60)
    safe_print("ЛАБОРАТОРНА РОБОТА 5")
    safe_print("Моніторинг ML моделі")
    safe_print("=" * 60)
    
    if not EVIDENTLY_AVAILABLE:
        safe_print("Evidently не встановлено!")
        return False
    
    try:
        engine = get_engine()
        safe_print("Підключено до бази даних")
        
        reference_df = load_reference_data(engine)
        if len(reference_df) == 0:
            safe_print("Не вдалося завантажити еталонні дані")
            return False
        
        current_df = load_current_data(engine)
        if len(current_df) == 0:
            safe_print("Не вдалося завантажити поточні дані")
            return False
        
        ref_prepared, curr_prepared = prepare_data(reference_df, current_df)
        column_mapping = create_column_mapping()
        
        success = generate_combined_dashboard(ref_prepared, curr_prepared, column_mapping)
        
        if success:
            safe_print("=" * 60)
            safe_print("Дашборд успішно згенеровано!")
            safe_print("Файл: lab5_monitoring_dashboard.html")
            safe_print("=" * 60)
            return True
        else:
            safe_print("\nНе вдалося згенерувати дашборд")
            return False
        
    except Exception as e:
        safe_print(f"\nПомилка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)