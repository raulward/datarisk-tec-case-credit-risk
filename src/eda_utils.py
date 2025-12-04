import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List


class DataUtils:
    """
    Classe utilitária para operações simples e recorrentes com dados.

    Atualmente fornece métodos para:

    - leitura de arquivos em formato CSV ou Excel (`.xlsx`);
    - centralização do caminho base de dados do projeto;
    - visualização univariada.

    A ideia é encapsular funções de uso geral relacionadas a dados
    em um único lugar, facilitando reutilização e manutenção.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def read_data(
        file: str,
        sep: str = ";",
        base_path: str = "../data",
        file_type: str = "csv",
        info: bool = False,
        processed: bool = False
    ) -> pd.DataFrame:
        """
        Lê um arquivo de dados e o retorna como um DataFrame do pandas.

        Esta função centraliza a lógica de leitura de arquivos,
        permitindo que o restante do código apenas informe o nome
        do arquivo e, opcionalmente, o tipo e o separador. Além disso,
        pode retornar informações a respeito do arquivo lido.

        Args:
            file (str):
                Nome do arquivo a ser lido. Ex.: "base_cadastral.csv".
            sep (str, opcional):
                Separador de colunas a ser utilizado na leitura de CSVs.
                Default = ';'.
            base_path (str, opcional):
                Caminho base onde os arquivos estão armazenados.
                Pode ser relativo ou absoluto. Default = "../data".
            file_type (str, opcional):
                Tipo do arquivo. Aceita:
                    - "csv"  → usa `pd.read_csv`
                    - "xlsx" → usa `pd.read_excel`
                Default = "csv".
            info (bool, opcional):
                Informações a respeito dos dados como:
                    - .info()
                    - Contagem de valores únicos e valores únicos
                    - Quantidade de registros e variáveis
                    - Quantidade de registros nulos

        Returns:
            pd.DataFrame:
                DataFrame contendo os dados lidos do arquivo.

        Raises:
            FileNotFoundError:
                Se o arquivo não for encontrado no caminho informado.
            ValueError:
                Se o `file_type` não for um dos tipos suportados.
        """
        base_path = Path(base_path)
        medium_path = 'processed' if processed else 'unprocessed'
        final_path = base_path / medium_path / file

        print(final_path)

        if not final_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {final_path}")

        if file_type == "csv":
            df = pd.read_csv(final_path, sep=sep)
        elif file_type == "xlsx":
            df = pd.read_excel(final_path)
        else:
            raise ValueError(
                f"Tipo de arquivo não suportado: {file_type}. "
                "Use 'csv' ou 'xlsx'."
            )

        if info:
            print('-' * 100)
            print()
            print('Shape')
            print()
            print(f'linhas: {df.shape[0]} | colunas: {df.shape[1]}')
            print()
            print('-' * 100)
            print()
            print('INFO')
            print()
            print(df.info())
            print()
            print('-' * 100)
            print()
            print('Valores únicos')
            print()
            for col in df.columns:
                print(f"\nColuna: {col}")
                print(df[col].nunique())
                if col != "ID_CLIENTE":
                    print(df[col].unique())
            print()
            print('-' * 100)
            print()
            print('Registros faltantes')
            print()
            df_missings = df.isna().sum().reset_index().rename(columns={'index': 'coluna', 0: 'Missings no.'}).sort_values('Missings no.')
            df_missings['Missing (%)'] = np.round((df_missings['Missings no.'] / df.shape[0]) * 100, 2)
            print(df_missings)
            print()
            print('-' * 100)



        return df

    @staticmethod
    def univariate_analysis_plots(
        data: pd.DataFrame,
        features: list,
        palette: str,
        histplot: bool = True,
        barplot: bool = False,
        mean: str | None = None,
        text_y: float = 0.5,
        kde: bool = False,
        figsize: tuple = (24, 12),
        clip_outliers: bool = True,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> None:
        """
        Gera gráficos para análise univariada de um conjunto de variáveis.

        A função adapta o tipo de visualização de acordo com o tipo da variável:

        - Variáveis numéricas:
            - Histograma (com porcentagem no eixo Y, opcionalmente com KDE)
            - Boxplot ao lado do histograma
            - Clipping automático de outliers por quantil, para evitar distorções visuais.

        - Variáveis categóricas:
            - Gráfico de barras horizontais mostrando a proporção de cada categoria;
            - Opcionalmente, gráfico de média de outra variável (`mean`) por categoria.

        Args:
            data (pd.DataFrame):
                DataFrame contendo os dados.
            features (list):
                Lista com os nomes das variáveis a serem visualizadas.
            histplot (bool, opcional):
                Se True, gera histogramas para variáveis numéricas. Default = True.
            barplot (bool, opcional):
                Se True, gera gráficos de barras horizontais para variáveis categóricas.
                Default = False.
            mean (str ou None, opcional):
                Se fornecido, o gráfico de barras categórico mostra a MÉDIA
                dessa variável ao invés da proporção. Default = None.
            text_y (float, opcional):
                Posição horizontal do texto sobre as barras nos gráficos categóricos.
                Default = 0.5.
            kde (bool, opcional):
                Se True, plota a curva KDE sobre os histogramas numéricos.
                Default = False.
            color (str, opcional):
                Cor principal dos gráficos. Default = "#8d0801".
            figsize (tuple, opcional):
                Tamanho total da figura (largura, altura). Default = (24, 12).
            clip_outliers (bool, opcional):
                Se True, realiza clipping de outliers em variáveis numéricas usando quantis.
                Default = True.
            lower_quantile (float, opcional):
                Quantil inferior para clipping (se clip_outliers=True). Default = 0.01.
            upper_quantile (float, opcional):
                Quantil superior para clipping (se clip_outliers=True). Default = 0.99.

        Returns:
            None
        """

        num_features = len(features)

        # se for só barplot (categóricas), 1 coluna é suficiente
        if barplot and not histplot:
            n_cols = 1
        else:
            n_cols = 2

        n_rows = num_features

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # normalizar axes para SEMPRE ser 2D (n_rows, n_cols)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, feature in enumerate(features):
            ax_left = axes[i, 0]
            ax_right = axes[i, 1] if n_cols == 2 else None

            series = data[feature].dropna()
            is_numeric = pd.api.types.is_numeric_dtype(series)

            # ---------------- VARIÁVEIS NUMÉRICAS ----------------
            if is_numeric and histplot:
                # Clipping de outliers para evitar gráficos “bizarros”
                if clip_outliers:
                    lo = series.quantile(lower_quantile)
                    hi = series.quantile(upper_quantile)
                    series_clipped = series.clip(lower=lo, upper=hi)
                else:
                    series_clipped = series

                sns.histplot(
                    x=series_clipped,
                    kde=kde,
                    stat="percent",
                    ax=ax_left,
                    palette=palette,
                    bins=30,
                )
                ax_left.set_title(f"{feature} - Histograma")
                ax_left.set_xlabel(feature)
                ax_left.set_ylabel("% de observações")
                ax_left.grid(False)

                if ax_right is not None:
                    sns.boxplot(
                        x=series,
                        ax=ax_right,
                        palette=palette,
                        orient="h",
                    )
                    ax_right.set_title(f"{feature} - Boxplot")
                    ax_right.set_xlabel("")
                    ax_right.set_yticks([])
                    ax_right.ticklabel_format(style='plain', axis='x')
                    ax_right.grid(False)

            # ---------------- VARIÁVEIS CATEGÓRICAS ----------------
            elif barplot:
                if mean is not None:
                    grouped = (
                        data.groupby(feature)[mean]
                        .mean()
                        .reset_index()
                        .sort_values(mean, ascending=True)
                    )
                    grouped[mean] = grouped[mean].round(2)

                    max_pct = grouped["pct"].max()
                    ax_left.set_xlim(0, max_pct * 1.15)

                    for idx, value in enumerate(grouped["pct"]):
                        ax_left.text(
                            x=value - 0.5,
                            y=idx,
                            s=f"{value:.1f}%",
                            va="center",
                            ha="right",
                            color="white",
                            fontsize=10,
                            fontweight="bold"
                        )

                    ax_left.set_xlabel(f"Média de {mean}")
                else:
                    grouped = (
                        series.value_counts(normalize=True)
                        .mul(100)
                        .rename("pct")
                        .reset_index()
                    )
                    grouped = grouped.rename(columns={"index": feature})
                    grouped = grouped.sort_values("pct", ascending=True)

                    ax_left.barh(
                        y=grouped[feature],
                        width=grouped["pct"],
                        pallete=palette,
                    )

                    for idx, value in enumerate(grouped["pct"]):
                        ax_left.text(
                            x=value + text_y,
                            y=idx,
                            s=f"{value:.1f}%",
                            va="center",
                            fontsize=10,
                        )

                    ax_left.set_xlabel("% de observações")

                ax_left.set_title(feature)
                ax_left.spines["top"].set_visible(False)
                ax_left.spines["right"].set_visible(False)
                ax_left.spines["bottom"].set_visible(False)
                ax_left.spines["left"].set_visible(False)
                ax_left.grid(False)

                if ax_right is not None:
                    ax_right.axis("off")

            # ---------------- CASO NÃO SE ENCAIXE ----------------
            else:
                ax_left.text(
                    0.5,
                    0.5,
                    f"Tipo de variável\nnão tratado\n({feature})",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
                ax_left.axis("off")
                if ax_right is not None:
                    ax_right.axis("off")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def woe_iv_table(
        data: pd.DataFrame,
        feature: str,
        target: str,
        bad_value: int | str = 1,
        bins: int | None = None,
        precision: int = 2,
    ) -> pd.DataFrame:
        """
        Gera tabela de análise de default para uma variável (WoE / IV).

        Args:
            data (pd.DataFrame): DataFrame com os dados.
            feature (str): Nome da variável explicativa a ser analisada.
            target (str): Nome da variável resposta binária (good/bad).
            bad_value (int | str, opcional): Valor que representa "mau pagador" no target.
                                             Tudo diferente disso será considerado "bom".
                                             Default = 1.
            bins (int | None, opcional): Se fornecido e a variável for numérica, será
                                         aplicado um qcut em `bins` faixas antes da análise.
                                         Default = None (usa categorias já existentes).
            precision (int, opcional): Casas decimais para arredondar percentuais/WoE/IV.
                                       Default = 2.

        Returns:
            pd.DataFrame: Tabela com n_obs, proporções, WoE e IV por categoria
                          e uma linha "total" com o IV total da variável.
        """

        df = data[[feature, target]].dropna().copy()

        # binarizar alvo: 1 = bad, 0 = good
        df["_bad"] = (df[target] == bad_value).astype(int)
        df["_good"] = 1 - df["_bad"]

        # tratar variável (binning para numérica, se solicitado)
        feat_col = feature
        if bins is not None and pd.api.types.is_numeric_dtype(df[feature]):
            df[feature + "_BIN"] = pd.qcut(df[feature], q=bins, duplicates="drop")
            feat_col = feature + "_BIN"

        # agregações por categoria
        grouped = df.groupby(feat_col, dropna=False)
        n_obs = grouped.size()
        n_good = grouped["_good"].sum()
        n_bad = grouped["_bad"].sum()

        total_good = n_good.sum()
        total_bad = n_bad.sum()

        # evitar divisão por zero
        eps = 1e-9

        good_row = (n_good / n_obs).replace([np.inf, -np.inf], np.nan)
        bad_row = (n_bad / n_obs).replace([np.inf, -np.inf], np.nan)

        good_col = (n_good / (total_good + eps)).replace([np.inf, -np.inf], np.nan)
        bad_col = (n_bad / (total_bad + eps)).replace([np.inf, -np.inf], np.nan)

        g_b_ratio = (good_col / (bad_col + eps)).replace([np.inf, -np.inf], np.nan)

        # WoE e IV por categoria
        woe = np.log((good_col + eps) / (bad_col + eps))
        iv = (good_col - bad_col) * woe

        # montar tabela
        table = pd.DataFrame({
            "n_obs": n_obs,
            "obs_proportion (%)": (n_obs / n_obs.sum()) * 100,
            "good_row (%)": good_row * 100,
            "bad_row (%)": bad_row * 100,
            "n_good": n_good,
            "n_bad": n_bad,
            "good_col (%)": good_col * 100,
            "bad_col (%)": bad_col * 100,
            "g/b": g_b_ratio,
            "woe": woe,
            "iv": iv,
        })

        # arredondar percentuais/WoE/IV
        cols_to_round = [
            "obs_proportion (%)",
            "good_row (%)",
            "bad_row (%)",
            "good_col (%)",
            "bad_col (%)",
            "g/b",
            "woe",
            "iv",
        ]
        table[cols_to_round] = table[cols_to_round].round(precision)

        # linha total
        total_row = pd.Series({
            "n_obs": n_obs.sum(),
            "obs_proportion (%)": 100.0,
            "good_row (%)": (total_good / n_obs.sum()) * 100,
            "bad_row (%)": (total_bad / n_obs.sum()) * 100,
            "n_good": total_good,
            "n_bad": total_bad,
            "good_col (%)": 100.0,
            "bad_col (%)": 100.0,
            "g/b": np.nan,
            "woe": np.nan,
            "iv": iv.sum().round(precision),
        }, name="total")

        table.index.name = f"{feature}_cat"
        table = pd.concat([table, total_row.to_frame().T])

        return table

    @staticmethod
    def plot_default_woe(
        data: pd.DataFrame,
        feature: str,
        target: str,
        palette: list,
        bad_value: int | str = 1,
        bins: int | None = None,
        figsize: tuple = (14, 4),
    ) -> None:
        tbl = DataUtils.woe_iv_table(
            data=data,
            feature=feature,
            target=target,
            bad_value=bad_value,
            bins=bins,
            precision=2,
        )
        tbl = tbl.loc[tbl.index != "total"].copy()
        tbl = tbl.sort_values("bad_row (%)", ascending=False)
        cats = tbl.index.astype(str)
        bad_pct = tbl["bad_row (%)"].values
        good_pct = tbl["good_row (%)"].values
        woe = tbl["woe"].values
        y_pos = np.arange(len(cats))
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax1, ax2 = axes
        ax1.barh(y_pos, bad_pct, color=palette[0], label="Bad")
        ax1.barh(y_pos, good_pct, left=bad_pct, color="#b0a9a9", label="Good")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(cats)
        ax1.invert_yaxis()
        ax1.grid(False)
        ax1.set_xlabel("% dentro da categoria")
        ax1.set_title(f"Default rate by {feature}")
        for i, (b, g) in enumerate(zip(bad_pct, good_pct)):
            ax1.text(b / 2, y_pos[i], f"{b:.2f}%", va="center", ha="center", color="white", fontsize=9)
            ax1.text(b + g / 2, y_pos[i], f"{g:.2f}%", va="center", ha="center", color="black", fontsize=9)
        ax1.legend(bbox_to_anchor=(-0.2, 1.1), loc='upper left')
        ax2.plot(cats, woe, marker="o", linestyle="--", color=palette[0])
        ax2.set_xlabel(feature)
        ax2.set_ylabel("WoE")
        ax2.set_title(f"WoE by {feature}")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def detect_outliers_iqr(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return series[(series < lower) | (series > upper)]

    @staticmethod
    def find_na_ocurrences_by_ids(
        id_cols: List[str],
        data: pd.DataFrame,
        features: List[str]
    ) -> pd.DataFrame:
        """
        Calcula, para cada feature em `features`:
        - média, mínimo e máximo da QUANTIDADE de NAs por ID (ou combinação de IDs);
        - média, mínimo e máximo da PORCENTAGEM de NAs por ID.

        Args:
            id_cols (List[str]): lista com o(s) nome(s) da(s) coluna(s) de ID
                                 (ex.: ['ID_CLIENTE']).
            data (pd.DataFrame): base de dados.
            features (List[str]): lista de colunas numéricas para analisar NAs.

        Returns:
            pd.DataFrame: tabela com métricas de NAs por feature.
        """

        results = []

        for feature in features:
            # agrupa por ID(s)
            grp = data.groupby(id_cols)[feature]

            # quantidade de NAs por ID
            na_count = grp.apply(lambda s: s.isna().sum())

            # total de registros por ID
            total_count = grp.size()

            # porcentagem de NAs por ID
            na_pct = na_count / total_count

            # monta dicionário de métricas
            results.append({
                "feature": feature,
                "mean_na_count": na_count.mean(),
                "max_na_count": na_count.max(),
                "min_na_count": na_count.min(),
                "mean_na_pct": na_pct.mean(),
                "max_na_pct": na_pct.max(),
                "min_na_pct": na_pct.min(),
            })

        return pd.DataFrame(results)
