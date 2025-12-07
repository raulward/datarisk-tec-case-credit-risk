from src.submission_utils import *
import pandas as pd

import joblib


def main():

    model = joblib.load("./credit_risk_pipeline.pkl")

    build_processed_parquets()

    df_test = prepare_test_dataframe(
        test_path="./data/unprocessed/base_pagamentos_teste.csv",
        file_type="csv",
        apply_feature_engineering=True,   
    )

    pd.set_option('display.max_columns', None)

    generate_submission(
        model=model,
        df_test=df_test,
        output_path="./submissao_case.csv",
        drop_ano_ref=True,  
    )

    print('Arquivo de submissão gerado gerado!')

if __name__ == "__main__":
    main()