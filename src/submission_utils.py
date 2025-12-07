# submission_utils.py

import numpy as np
import pandas as pd
from typing import Tuple

import os

from src.modelling_utils import read_and_clean_data, feature_engineering

from pathlib import Path
from src.eda_utils import read_data


COLS_TO_DROP = [
    "DATA_EMISSAO_DOCUMENTO",
    "DATA_PAGAMENTO",
    "DATA_VENCIMENTO",
    "DDD",
    "CEP_2_DIG",
    "DATA_CADASTRO",
    "DOMINIO_EMAIL",
]


def _add_date_and_relationship_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replica as criações de features temporais usadas no notebook de modelagem:
    - DIA_VENCIMENTO_DOCUMENTO
    - MES_VENCIMENTO_DOCUMENTO
    - TEMPO_DE_PAGAMENTO
    - DIA_REF, MES_REF, ANO_REF
    - TEMPO_RELACIONAMENTO
    """
    df = df.copy()

    df["DIA_VENCIMENTO_DOCUMENTO"] = df["DATA_VENCIMENTO"].dt.day
    df["MES_VENCIMENTO_DOCUMENTO"] = df["DATA_VENCIMENTO"].dt.month
    df["TEMPO_DE_PAGAMENTO"] = (df["DATA_VENCIMENTO"] - df["DATA_EMISSAO_DOCUMENTO"]).dt.days

    df["DIA_REF"] = df["SAFRA_REF"].dt.day
    df["MES_REF"] = df["SAFRA_REF"].dt.month

    df["TEMPO_RELACIONAMENTO"] = (
        (df["SAFRA_REF"].dt.year - df["DATA_CADASTRO"].dt.year) * 12
        + (df["SAFRA_REF"].dt.month - df["DATA_CADASTRO"].dt.month)
    )

    return df


def _drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replica o drop de colunas feito no notebook de modelagem.
    """
    df = df.copy()
    cols_existing = [c for c in COLS_TO_DROP if c in df.columns]
    return df.drop(columns=cols_existing)


def prepare_development_dataframe(
    dev_path: str = "../data/processed/base_pagamentos_desenvolvimento.xlsx",
    file_type: str = "xlsx",
) -> pd.DataFrame:
    """
    Lê e trata a base de desenvolvimento, reproduzindo o que foi feito
    no notebook de modelagem, até o dataframe final SEM feature engineering.
    """
    df = read_and_clean_data(dev_path, file_type=file_type)
    df = _add_date_and_relationship_features(df)
    df = _drop_unused_columns(df)
    return df


def prepare_development_dataframe_fe(
    dev_path: str = "../data/processed/base_pagamentos_desenvolvimento.xlsx",
    file_type: str = "xlsx",
) -> pd.DataFrame:
    """
    Versão com feature engineering adicional, conforme notebook:
    - aplica prepare_development_dataframe
    - aplica feature_engineering (DTI_FATURAMENTO, PAYMENT_SHOCK, histórico de inadimplência etc.)
    """
    df = prepare_development_dataframe(dev_path=dev_path, file_type=file_type)
    df_fe = feature_engineering(df)
    return df_fe


def timed_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replica o split usado no notebook
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size deve estar entre 0 e 1.")

    split_idx = int((1 - test_size) * len(df))
    train, test = np.split(df, [split_idx])

    train = train.copy()
    test = test.copy()
    train["SET"] = "train"
    test["SET"] = "test"

    return train, test


def get_X_y_from_dataframe(
    df: pd.DataFrame,
    drop_ano_ref: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Replica a construção de X e y do notebook.

    Baseline:
        X = df.drop(['TARGET', 'ID_CLIENTE', 'SAFRA_REF'], axis=1)
        y = df['TARGET']

    FE:
        X = df.drop(['TARGET', 'ID_CLIENTE', 'SAFRA_REF', 'ANO_REF'], axis=1)
        y = df['TARGET']
    """
    drop_cols = ["TARGET", "ID_CLIENTE", "SAFRA_REF"]
    if drop_ano_ref and "ANO_REF" in df.columns:
        drop_cols.append("ANO_REF")

    X = df.drop(columns=drop_cols)
    y = df["TARGET"].astype(int)

    return X, y


def _build_cadastral_and_info(
    base_cadastral_path: str = "../data/processed/base_cadastral.parquet",
    base_info_path: str = "../data/processed/base_info.parquet",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constrói as bases cadastral e info com os MESMOS tratamentos usados em read_and_clean_data,
    para reaproveitar com a base de teste.
    """
    bc_df = pd.read_parquet(base_cadastral_path)
    bi_df = pd.read_parquet(base_info_path)

    # REGIAO_DDD
    mask_valid_bc_ddd = bc_df["DDD"] != "INVÁLIDO"
    bc_df["REGIAO_DDD"] = np.where(
        mask_valid_bc_ddd,
        bc_df["DDD"].astype(str).str[0],
        np.nan,
    )

    # REGIAO_CEP
    cep_str = (
        bc_df["CEP_2_DIG"]
        .astype("Int64")
        .astype(str)
        .where(bc_df["CEP_2_DIG"].notna(), other=np.nan)
    )
    mask_valid_cep = cep_str.notna()

    bc_df["REGIAO_CEP"] = np.where(
        mask_valid_cep,
        cep_str.str[0],
        np.nan,
    )

    # Tratamentos em base_info
    bi_df["SAFRA_REF"] = pd.to_datetime(bi_df["SAFRA_REF"])
    bi_df = bi_df.sort_values(["ID_CLIENTE", "SAFRA_REF"])

    cols_to_impute_bi = ["NO_FUNCIONARIOS", "RENDA_MES_ANTERIOR"]
    for col in cols_to_impute_bi:
        bi_df[col] = bi_df.groupby("ID_CLIENTE")[col].ffill()

    return bc_df, bi_df

def prepare_test_dataframe(
    test_path: str = "./data/unprocessed/base_pagamentos_teste.csv",
    file_type: str = "csv",
    base_cadastral_path: str = "./data/processed/base_cadastral.parquet",
    base_info_path: str = "./data/processed/base_info.parquet",
    apply_feature_engineering: bool = True,
) -> pd.DataFrame:
    """
    Prepara a base de teste (base_pagamentos_teste), espelhando o fluxo da base de desenvolvimento,
    exceto pela criação da TARGET.

    Etapas (espelhando read_and_clean_data + notebook):
    - leitura da base de pagamentos de teste
    - conversão das colunas de data
    - sort por SAFRA_REF
    - forward-fill de VALOR_A_PAGAR por ID_CLIENTE
    - construção de REGIAO_DDD / REGIAO_CEP e imputações em base_info
    - merge com base_cadastral e base_info
    - criação de features de data / relacionamento
    - drop das colunas descartadas
    - (opcional) feature_engineering
    """
    # leitura base de pagamentos de teste
    if file_type == "csv":
        test_df = pd.read_csv(test_path, sep=";")
    elif file_type == "parquet":
        test_df = pd.read_parquet(test_path)
    else:
        raise ValueError("file_type deve ser 'csv' ou 'parquet'")

    # datas
    test_df["SAFRA_REF"] = pd.to_datetime(test_df["SAFRA_REF"])
    test_df["DATA_EMISSAO_DOCUMENTO"] = pd.to_datetime(test_df["DATA_EMISSAO_DOCUMENTO"])
    test_df["DATA_VENCIMENTO"] = pd.to_datetime(test_df["DATA_VENCIMENTO"])

    # ordenação temporal
    test_df = test_df.sort_values("SAFRA_REF")

    
    if "VALOR_A_PAGAR" in test_df.columns:
        test_df["VALOR_A_PAGAR"] = (
            test_df
            .groupby("ID_CLIENTE")["VALOR_A_PAGAR"]
            .ffill()
        )

    bc_df, bi_df = _build_cadastral_and_info(
        base_cadastral_path=base_cadastral_path,
        base_info_path=base_info_path,
    )

    final_df = (
        test_df
        .merge(bc_df, on="ID_CLIENTE", how="left")
        .merge(bi_df, on=["ID_CLIENTE", "SAFRA_REF"], how="left")
    )

    final_df = _add_date_and_relationship_features(final_df)
    final_df = _drop_unused_columns(final_df)

    if apply_feature_engineering:
        df_for_fe = final_df.copy()

        if 'TARGET' not in df_for_fe.columns:
            df_for_fe['TARGET'] = 0

        df_for_fe = feature_engineering(df_for_fe)

        if 'TARGET' in df_for_fe.columns and 'TARGET' not in final_df.columns:
            df_for_fe = df_for_fe.drop(columns=['TARGET'])

        final_df = df_for_fe

    return final_df

def generate_submission(
    model,
    df_test: pd.DataFrame,
    output_path: str = "../submissao_case.csv",
    drop_ano_ref: bool = True,
) -> pd.DataFrame:
    """
    Gera o arquivo de submissão a partir de:
      - um modelo já treinado (tipicamente um Pipeline com preprocess + clf)
      - dataframe de teste já preparado por prepare_test_dataframe

    X_test replica o que foi feito na modelagem:
      - se drop_ano_ref=True: drop ['TARGET', 'ID_CLIENTE', 'SAFRA_REF', 'ANO_REF'] (se TARGET existir)
      - se drop_ano_ref=False: drop ['TARGET', 'ID_CLIENTE', 'SAFRA_REF']
    """
    drop_cols = ["ID_CLIENTE", "SAFRA_REF"]
    if "TARGET" in df_test.columns:
        drop_cols.append("TARGET")
    if drop_ano_ref and "ANO_REF" in df_test.columns:
        drop_cols.append("ANO_REF")

    X_test = df_test.drop(columns=drop_cols)

    # o modelo deve possuir predict_proba
    if not hasattr(model, "predict_proba"):
        raise AttributeError("O modelo informado não possui método 'predict_proba'.")

    y_proba = model.predict_proba(X_test)[:, 1]

    submission = df_test[["ID_CLIENTE", "SAFRA_REF"]].copy()
    submission["PROBABILIDADE_INADIMPLENCIA"] = y_proba

    submission.to_csv(output_path, index=False)

    return submission

def build_processed_parquets(
    base_path: str = "./data/",
    base_cadastral_file: str = "base_cadastral.csv",
    base_pagamento_file: str = "base_pagamentos_desenvolvimento.csv",
    base_info_file: str = "base_info.csv",
    sep: str = ";",
) -> None:
    """
    Lê os arquivos brutos da pasta `unprocessed` e salva em formato parquet
    na pasta `processed`, fazendo apenas as conversões de data necessárias
    para o fluxo de modelagem.

    """

    processed_dir = base_path + "processed/"
    os.makedirs(processed_dir, exist_ok=True)

    bc_df = pd.read_csv(base_path + "unprocessed/" + base_cadastral_file, sep=sep)

    if "DATA_CADASTRO" in bc_df.columns:
        bc_df["DATA_CADASTRO"] = pd.to_datetime(bc_df["DATA_CADASTRO"])
    
    bc_df.loc[bc_df['CEP_2_DIG'] == 'na', 'CEP_2_DIG'] = np.nan

    bc_df.to_parquet(processed_dir + "base_cadastral.parquet", index=False)


    bi_df = pd.read_csv(base_path + "unprocessed/" + base_info_file, sep=sep)


    if "SAFRA_REF" in bi_df.columns:
        bi_df["SAFRA_REF"] = pd.to_datetime(bi_df["SAFRA_REF"])

    bi_df.to_parquet(processed_dir + "base_info.parquet", index=False)

    dev_def = pd.read_csv(base_path + "unprocessed/" + base_pagamento_file, sep=sep)

    datetime_cols = ['SAFRA_REF', 'DATA_EMISSAO_DOCUMENTO', 'DATA_PAGAMENTO', 'DATA_VENCIMENTO']

    dev_def[datetime_cols] = dev_def[datetime_cols].apply(pd.to_datetime)

    dev_def.to_parquet(processed_dir + "base_pagamento_desenvolvimentos.parquet", index=False)

    print("Parquets gerados em:", processed_dir)
