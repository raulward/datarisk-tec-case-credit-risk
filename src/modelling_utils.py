import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, average_precision_score, precision_recall_curve

from warnings import filterwarnings
filterwarnings('ignore')


def read_and_clean_data(dev_path: str, file_type: str) -> pd.DataFrame:
    """
    Le e limpa base
    """
    base_info_path = '../data/processed/base_info.parquet'
    base_cad_path = '../data/processed/base_cadastral.parquet'

    if file_type == 'csv':
        dev_df = pd.read_csv(dev_path)
    elif file_type == 'parquet':
        dev_df = pd.read_parquet(dev_path)
    else:
        raise ValueError("file_type must be 'csv' or 'parquet'")

    dev_df['SAFRA_REF'] = pd.to_datetime(dev_df['SAFRA_REF'])
    dev_df['DATA_PAGAMENTO'] = pd.to_datetime(dev_df['DATA_PAGAMENTO'])
    dev_df['DATA_VENCIMENTO'] = pd.to_datetime(dev_df['DATA_VENCIMENTO'])

    dev_df.sort_values('SAFRA_REF', inplace=True)

    bc_df = pd.read_parquet(base_cad_path)
    bi_df = pd.read_parquet(base_info_path)

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
    """
    Calcula KS
    """
    scores_bad = y_score[y_true == 1]
    scores_good = y_score[y_true == 0]

    values = np.sort(np.unique(y_score))

    cdf_bad = np.searchsorted(np.sort(scores_bad), values, side='right') / scores_bad.size
    cdf_good = np.searchsorted(np.sort(scores_good), values, side='right') / scores_good.size

    ks = np.max(np.abs(cdf_bad - cdf_good))
    return ks

def gini_coefficient(y_true, y_score):
    """
    Calcula o GINI
    """
    auc = roc_auc_score(y_true, y_score)
    return 2 * auc - 1

def evaluate_global_metrics(y_true, y_score):
    """
    Avalia as principais métricas de desempenho
    """
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
    """
    Retorna a importancia de cada feature para a regressão linear
    """
    clf = pipeline.named_steps['clf']
    coef = clf.coef_[0]

    imp = pd.DataFrame({
        'feature': feature_names,
        'coef': coef
    })
    imp['abs_coef'] = imp['coef'].abs()
    imp = imp.sort_values('abs_coef', ascending=False)
    return imp

def recencia_inad_serie(s: pd.Series) -> pd.Series:
    """
    Feature para calcular a recencia de inadimplencia
    """
    rec = []
    last_bad = None
    shifted = s.shift(1)

    for i, v in enumerate(shifted):
        if last_bad is None:
            rec.append(np.nan)
        else:
            rec.append(i - last_bad)
        if v == 1:
            last_bad = i

    return pd.Series(rec, index=s.index)

def plot_roc_curve(y_true, y_score, title='Curva ROC'):
    """
    Plota curva ROC
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Aleatório')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def feature_engineering(df):
    """
    Cria features de crédito, incluindo razões financeiras,
    volatilidade e histórico de inadimplência agregado por safra
    (evitando data leakage intra-mês).
    """

    df_feat = df.copy()

    if 'ID_CLIENTE' in df_feat.columns and 'SAFRA_REF' in df_feat.columns:
        df_feat = df_feat.sort_values(['ID_CLIENTE', 'SAFRA_REF'])

    df_feat['DTI_FATURAMENTO'] = df_feat['VALOR_A_PAGAR'] / df_feat['RENDA_MES_ANTERIOR']

    df_feat['TICKET_POR_FUNC'] = df_feat['VALOR_A_PAGAR'] / df_feat['NO_FUNCIONARIOS']

    df_feat['MEDIA_VALOR_3M'] = df_feat.groupby('ID_CLIENTE')['VALOR_A_PAGAR'].transform(
        lambda x: x.shift(1).rolling(window=3).mean()
    )

    df_feat['PAYMENT_SHOCK'] = df_feat['VALOR_A_PAGAR'] / df_feat['MEDIA_VALOR_3M']

    cols_to_clean = ['DTI_FATURAMENTO', 'TICKET_POR_FUNC', 'PAYMENT_SHOCK', 'MEDIA_VALOR_3M']
    df_feat[cols_to_clean] = df_feat[cols_to_clean].replace([np.inf, -np.inf], np.nan)
    df_feat[cols_to_clean] = df_feat[cols_to_clean].fillna(0)

    hist_safra = df_feat.groupby(['ID_CLIENTE', 'SAFRA_REF'])['TARGET'].max().reset_index()
    hist_safra = hist_safra.sort_values(['ID_CLIENTE', 'SAFRA_REF'])

    hist_safra['HIST_INAD_ANTERIOR'] = hist_safra.groupby('ID_CLIENTE', group_keys=False)['TARGET'] \
                                                 .apply(lambda x: x.shift(1).cumsum())

    hist_safra['HIST_FREQ_3M'] = hist_safra.groupby('ID_CLIENTE', group_keys=False)['TARGET'] \
                                           .apply(lambda x: x.shift(1).rolling(3).mean())

    hist_safra = hist_safra.drop(columns=['TARGET'])
    df_feat = df_feat.merge(hist_safra, on=['ID_CLIENTE', 'SAFRA_REF'], how='left')

    df_feat[['HIST_INAD_ANTERIOR', 'HIST_FREQ_3M']] = df_feat[['HIST_INAD_ANTERIOR', 'HIST_FREQ_3M']].fillna(0)

    df_feat['N_COBRANCAS'] = df_feat.groupby('ID_CLIENTE').cumcount()

    return df_feat

def compare_model_variants(experiments,
                           plot_roc=True,
                           plot_pr=True,
                           plot_bar=True,
                           title_roc="Curvas ROC - Comparação de Modelos"):
    """
    Compara variantes de modelos que usam CONJUNTOS DE DADOS DIFERENTES.
    """

    results = []
    y_scores_dict = {}

    for name, cfg in experiments.items():
        model = cfg["model"]
        X_train = cfg["X_train"]
        y_train = cfg["y_train"]
        X_valid = cfg["X_valid"]
        y_valid = pd.Series(cfg["y_valid"]).astype(int)

        # treina no dataset específico da variante
        model.fit(X_train, y_train)

        # probas
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_valid)[:, 1]
        else:
            y_score = model.decision_function(X_valid)

        y_scores_dict[name] = (y_valid, y_score)

        roc_auc = roc_auc_score(y_valid, y_score)
        avg_prec = average_precision_score(y_valid, y_score)
        brier = brier_score_loss(y_valid, y_score)

        results.append({
            "modelo": name,
            "ROC_AUC": roc_auc,
            "GINI": 2 * roc_auc - 1,
            "Average Precision": avg_prec,
            "Brier Score": brier,
        })

    results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)

    # ----------------- ROC -----------------
    if plot_roc:
        plt.figure(figsize=(10, 6))
        for name, (y_valid, y_score) in y_scores_dict.items():
            fpr, tpr, _ = roc_curve(y_valid, y_score)
            auc_val = roc_auc_score(y_valid, y_score)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")

        plt.plot([0, 1], [0, 1], linestyle="--", label="Aleatório")
        plt.xlabel("Falso Positivo (FPR)")
        plt.ylabel("Verdadeiro Positivo (TPR)")
        plt.title(title_roc)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ------------- Precision–Recall ---------
    if plot_pr:
        plt.figure(figsize=(10, 6))
        for name, (y_valid, y_score) in y_scores_dict.items():
            precision, recall, _ = precision_recall_curve(y_valid, y_score)
            ap = average_precision_score(y_valid, y_score)
            plt.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")

        plt.xlabel("Recall")
        plt.ylabel("Precisão")
        plt.title("Curvas Precision-Recall - Comparação de Modelos")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ------------- Barras de métricas -------
    if plot_bar:
        plt.figure(figsize=(8, 5))
        metrics_plot = results_df.melt(
            id_vars="modelo",
            value_vars=["ROC_AUC", "GINI", "Brier Score"],
            var_name="métrica",
            value_name="valor"
        )
        ax = sns.barplot(data=metrics_plot, x="métrica", y="valor", hue="modelo")
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2., height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points"
            )
        plt.title("Comparação de Métricas por Variante")
        plt.tight_layout()
        plt.show()

    return results_df, y_scores_dict
