import pandas as pd
import numpy as np
import joblib

def preparar_base_consolidada(df_pagamentos, df_cadastral, df_info, is_train=True):
    """
    Aplica todas as transformações de limpeza e junção de dados.
    Pode receber DataFrames ou caminhos de arquivo.
    """

    if isinstance(df_pagamentos, str): df_pagamentos = pd.read_csv(df_pagamentos) if df_pagamentos.endswith('.csv') else pd.read_excel(df_pagamentos)
    if isinstance(df_cadastral, str): df_cadastral = pd.read_csv(df_cadastral) if df_cadastral.endswith('.csv') else pd.read_excel(df_cadastral)
    if isinstance(df_info, str): df_info = pd.read_csv(df_info) if df_info.endswith('.csv') else pd.read_excel(df_info)

    cols_data = ['SAFRA_REF', 'DATA_EMISSAO_DOCUMENTO', 'DATA_PAGAMENTO', 'DATA_VENCIMENTO']
    for col in cols_data:
        if col in df_pagamentos.columns:
            df_pagamentos[col] = pd.to_datetime(df_pagamentos[col])

    df_pagamentos.sort_values(['ID_CLIENTE', 'SAFRA_REF'], inplace=True)

    if 'FLAG_PF' in df_cadastral.columns:
        df_cadastral['FLAG_PF'] = df_cadastral['FLAG_PF'].fillna(0).replace('X', 1).astype(int)

    if 'DATA_CADASTRO' in df_cadastral.columns:
        df_cadastral['DATA_CADASTRO'] = pd.to_datetime(df_cadastral['DATA_CADASTRO'])

    if 'DDD' in df_cadastral.columns:
        df_cadastral['REGIAO_DDD'] = pd.to_numeric(df_cadastral['DDD'], errors='coerce').fillna(0).astype(int).astype(str).str[0]

    if 'CEP_2_DIG' in df_cadastral.columns:
        df_cadastral['REGIAO_CEP'] = pd.to_numeric(df_cadastral['CEP_2_DIG'], errors='coerce').fillna(0).astype(int).astype(str).str[0]

    if 'SAFRA_REF' in df_info.columns:
        df_info['SAFRA_REF'] = pd.to_datetime(df_info['SAFRA_REF'])

    df_info = df_info.sort_values(['ID_CLIENTE', 'SAFRA_REF'])
    cols_impute = ['NO_FUNCIONARIOS', 'RENDA_MES_ANTERIOR']
    for col in cols_impute:
        if col in df_info.columns:
            df_info[col] = df_info.groupby('ID_CLIENTE')[col].ffill()

    df_final = df_pagamentos.merge(df_cadastral, on='ID_CLIENTE', how='left')
    df_final = df_final.merge(df_info, on=['ID_CLIENTE', 'SAFRA_REF'], how='left')

    if is_train and 'DATA_PAGAMENTO' in df_final.columns and 'DATA_VENCIMENTO' in df_final.columns:
        df_final['DIAS_ATRASO'] = (df_final['DATA_PAGAMENTO'] - df_final['DATA_VENCIMENTO']).dt.days
        df_final['TARGET'] = np.where(df_final['DIAS_ATRASO'] >= 5, 1, 0)

    return df_final

def aplicar_feature_engineering(df):
    """
    Cria as variáveis derivadas para o modelo.
    Espera um DataFrame ordenado por ID_CLIENTE e SAFRA_REF.
    """
    df = df.copy()

    df = df.sort_values(['ID_CLIENTE', 'SAFRA_REF'])

    if 'DATA_VENCIMENTO' in df.columns:
        df['DIA_VENCIMENTO_DOCUMENTO'] = df['DATA_VENCIMENTO'].dt.day
        df['MES_VENCIMENTO_DOCUMENTO'] = df['DATA_VENCIMENTO'].dt.month

    if 'SAFRA_REF' in df.columns:
        df['DIA_REF'] = df['SAFRA_REF'].dt.day
        df['MES_REF'] = df['SAFRA_REF'].dt.month
        df['ANO_REF'] = df['SAFRA_REF'].dt.year

    if 'DATA_CADASTRO' in df.columns and 'SAFRA_REF' in df.columns:
        df['TEMPO_RELACIONAMENTO'] = (
            (df['SAFRA_REF'].dt.year - df['DATA_CADASTRO'].dt.year) * 12 +
            (df['SAFRA_REF'].dt.month - df['DATA_CADASTRO'].dt.month)
        )

    if 'VALOR_A_PAGAR' in df.columns and 'RENDA_MES_ANTERIOR' in df.columns:
        df['DTI_FATURAMENTO'] = df['VALOR_A_PAGAR'] / df['RENDA_MES_ANTERIOR'].replace(0, np.nan)

    if 'VALOR_A_PAGAR' in df.columns and 'NO_FUNCIONARIOS' in df.columns:
        df['TICKET_POR_FUNC'] = df['VALOR_A_PAGAR'] / df['NO_FUNCIONARIOS'].replace(0, np.nan)

    df['MEDIA_VALOR_3M'] = df.groupby('ID_CLIENTE')['VALOR_A_PAGAR'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )

    df['PAYMENT_SHOCK'] = df['VALOR_A_PAGAR'] / df['MEDIA_VALOR_3M'].replace(0, np.nan)

    df['N_COBRANCAS'] = df.groupby('ID_CLIENTE').cumcount()

    df['HIST_FREQ_3M'] = df.groupby('ID_CLIENTE')['VALOR_A_PAGAR'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).count()
    )

    if 'TARGET' in df.columns:
        df['HIST_INAD_ANTERIOR'] = df.groupby('ID_CLIENTE')['TARGET'].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).max()
        ).fillna(0)
    else:

        df['HIST_INAD_ANTERIOR'] = 0

    cols_to_fill = ['DTI_FATURAMENTO', 'TICKET_POR_FUNC', 'MEDIA_VALOR_3M', 'PAYMENT_SHOCK']
    for col in cols_to_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df

def gerar_previsoes_submissao(df_pagamentos_novo, df_cadastral, df_info):

    pipeline_carregado = joblib.load('../credit_risk_pipeline.pkl')

    df_sub = preparar_base_consolidada(df_pagamentos_novo, df_cadastral, df_info, is_train=False)

    df_sub_fe = aplicar_feature_engineering(df_sub)

    probas = pipeline_carregado.predict_proba(df_sub_fe)[:, 1]

    return probas
