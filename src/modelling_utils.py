# Data manipulation and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Modeling.
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
import statsmodels.api as sm

import sys
import os
import pickle

from warnings import filterwarnings
filterwarnings('ignore')


def read_and_clean_data(dev_path: str, file_type: str) -> pd.DataFrame:
    base_info_path = '../data/processed/base_info.xlsx'
    base_cad_path = '../data/processed/base_cadastral.xlsx'

    if file_type == 'csv':
        dev_df = pd.read_csv(dev_path)
    elif file_type == 'xlsx':
        dev_df = pd.read_excel(dev_path)
    else:
        raise ValueError("file_type must be 'csv' or 'xlsx'")

    dev_df['SAFRA_REF'] = pd.to_datetime(dev_df['SAFRA_REF'])
    dev_df['DATA_PAGAMENTO'] = pd.to_datetime(dev_df['DATA_PAGAMENTO'])
    dev_df['DATA_VENCIMENTO'] = pd.to_datetime(dev_df['DATA_VENCIMENTO'])

    dev_df.sort_values('SAFRA_REF', inplace=True)

    bc_df = pd.read_excel(base_cad_path)
    bi_df = pd.read_excel(base_info_path)

    dev_df['DIAS_ATRASO'] = (dev_df['DATA_PAGAMENTO'] - dev_df['DATA_VENCIMENTO']).dt.days
    dev_df['TARGET'] = np.where(dev_df['DIAS_ATRASO'] >= 5, 1, 0)
    dev_df.drop('DIAS_ATRASO', axis=1, inplace=True)

    dev_df['VALOR_A_PAGAR'] = (
        dev_df
        .groupby('ID_CLIENTE')['VALOR_A_PAGAR']
        .ffill()
    )

    mask_valid_bc_ddd = bc_df['DDD'] != 'INVÁLIDO'

    bc_df['REGIAO_DDD'] = np.where(
        mask_valid_bc_ddd,
        bc_df['DDD'].astype(str).str[0],
        np.nan
    )

    cep_str = (
        bc_df['CEP_2_DIG']
        .astype('Int64')
        .astype(str)
        .where(bc_df['CEP_2_DIG'].notna(), other=np.nan)
    )
    mask_valid_cep = cep_str.notna()

    bc_df['REGIAO_CEP'] = np.where(
        mask_valid_cep,
        cep_str.str[0],
        np.nan
    )

    bi_df['SAFRA_REF'] = pd.to_datetime(bi_df['SAFRA_REF'])
    bi_df = bi_df.sort_values(['ID_CLIENTE', 'SAFRA_REF'])

    cols_to_impute_bi = ['NO_FUNCIONARIOS', 'RENDA_MES_ANTERIOR']

    for col in cols_to_impute_bi:
        bi_df[col] = (
            bi_df
            .groupby('ID_CLIENTE')[col]
            .ffill()
        )

    final_df = (
        dev_df
        .merge(bc_df, on='ID_CLIENTE', how='left')
        .merge(bi_df, on=['ID_CLIENTE', 'SAFRA_REF'], how='left')
    )

    return final_df


def ks_statistic(y_true, y_score):
    scores_bad = y_score[y_true == 1]
    scores_good = y_score[y_true == 0]

    values = np.sort(np.unique(y_score))

    cdf_bad = np.searchsorted(np.sort(scores_bad), values, side='right') / scores_bad.size
    cdf_good = np.searchsorted(np.sort(scores_good), values, side='right') / scores_good.size

    ks = np.max(np.abs(cdf_bad - cdf_good))
    return ks


def gini_coefficient(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    return 2 * auc - 1


def evaluate_global_metrics(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    ks = ks_statistic(y_true, y_score)
    gini = gini_coefficient(y_true, y_score)
    brier = brier_score_loss(y_true, y_score)

    return {
        "AUC": auc,
        "KS": ks,
        "Gini": gini,
        "Brier": brier
    }

def get_logreg_importance(pipeline, feature_names):
    clf = pipeline.named_steps['clf']
    coef = clf.coef_[0]

    imp = pd.DataFrame({
        'feature': feature_names,
        'coef': coef
    })
    imp['abs_coef'] = imp['coef'].abs()
    imp = imp.sort_values('abs_coef', ascending=False)
    return imp
