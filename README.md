# 📘 README – Projeto de Previsão de Inadimplência

### Case Técnico – Cientista de Dados Júnior · Datarisk

### Probabilidade de Inadimplência em Cobranças Mensais

## **1. Visão Geral do Projeto**

Este projeto tem como objetivo desenvolver um modelo preditivo capaz de estimar a **probabilidade de inadimplência** dos clientes da empresa, considerando suas características cadastrais, comportamento mensal e histórico de pagamentos.

A inadimplência foi definida conforme regra do case:

> **Um pagamento é considerado inadimplente se for realizado com 5 dias ou mais de atraso em relação à data de vencimento.**

As previsões finais são geradas sobre a base `base_pagamentos_teste.csv`, produzindo o arquivo final:

```
submissao_case.csv
├── ID_CLIENTE
├── SAFRA_REF
└── PROBABILIDADE_INADIMPLENCIA
```

**Versão do python utilizada**: 3.13.7

## **Para executar o projeto**

Navegue até a pasta onde o projeto está instalado, então:

### **1. Criar o ambiente virtual**

```bash
python -m venv venv
```

---

### **2. Ativar o ambiente virtual**

#### **Windows (PowerShell)**

```bash
venv\Scripts\Activate
```

#### **Windows (CMD)**

```cmd
venv\Scripts\activate.bat
```

#### **Windows (Git Bash)**

```bash
source venv/Scripts/activate
```

#### **Linux / macOS**

```bash
source venv/bin/activate
```

---

### **3. Instalar as dependências**

Certifique-se de estar dentro do ambiente virtual ativado.

```bash
pip install -r requirements.txt
```

---

### **4. Estrutura esperada de diretórios**

Certifique-se de que o projeto contenha a seguinte estrutura:

```
project/
│
├── data/
│   ├── base_cadastral.csv
│   ├── base_info.csv
│   ├── base_pagamentos_desenvolvimento.csv
│   └── base_pagamentos_teste.csv
│
├── notebooks/
│   ├── 1_data_understanding.ipynb
│   ├── 2_eda.ipynb
│   └── 3_modelling.ipynb
│
├── src/
│   ├── data_utils.py
│   ├── eda_utils.py
│   ├── modelling_utils.py
│   └── submission_utils.py
│
├── assets/     ← gráficos gerados automaticamente
│
├── README.md
├── requirements.txt
├── README.md  (este arquivo)
├── submissao_case.csv ← arquivo final de previsão
└── main.py
  
```

> Ao final da execução, o arquivo **submissao_case.csv** será criado automaticamente.
**5. Reproduzir a submissão**

Se desejar gerar apenas o arquivo final, sem rodar os notebooks:

```bash
python ./main.py
```

**6. Executar o pipeline de modelagem**

O fluxo típico consiste em rodar os notebooks em ordem:

1. **1_data_understanding.ipynb**
   Carrega e inspeciona bases, verifica integridade e tipos.

2. **2_eda.ipynb**
   Exploração completa dos dados, gráficos, WOE, análises temporais.

3. **3_modelling.ipynb**

   * Constrói a base final de modelagem
   * Treina os modelos
   * Avalia desempenho

## **2. Bases Utilizadas**

Foram disponibilizadas quatro bases, vinculadas pelas chaves `ID_CLIENTE` e `SAFRA_REF`.

### **2.1 base_pagamentos_desenvolvimento**

Histórico de cobranças, usado para:
* Construção da variável target
* Feature engineering comportamental
* Treinamento e validação do modelo

### **2.2 base_pagamentos_teste**

Registros mais recentes, sem informação de pagamento.
É sobre essa base que o modelo prevê.

### **2.3 base_cadastral**

Informações cadastrais fixas por cliente:
* DDD
* Segmento industrial
* Dominio do e-mail
* Porte
* CEP_2_DIG
* FLAG_PF

### **2.4 base_info**

Informações mensais variáveis:
* renda do mês anterior
* número de funcionários

## **3. Construção da Variável Target**

Para cada linha:

```python
dias_atraso = (DATA_PAGAMENTO - DATA_VENCIMENTO).dt.days
target = (dias_atraso >= 5).astype(int)
```

Casos sem pagamento foram corretamente tratados como:
* Se DATA_PAGAMENTO está ausente, então inadimplente (`target=1`)
* Para a base de teste, target não é criado

## **4. Exploração de Dados (EDA) – Principais Insights**

A EDA foi dividida em:

* Análise da base de pagamentos
* Análise cadastral
* Análise de informações mensais
* Comportamento ao longo do tempo

A seguir, os **insights mais relevantes**.

### **4.1 Sazonalidade e comportamento temporal**

Gráficos mostraram:

* A inadimplência aumenta em determinados meses (provavelmente sazonalidade financeira).
* Distribuição de atrasos tem cauda longa: muitos clientes pagam poucos dias atrasados, mas há clusters de atrasos severos.

### **4.2 Correlação e variáveis mais relevantes**

A matriz de correlação (assets `matriz_corr_eda.png`) destacou:

* `DIAS_ATRASO` (apenas para desenvolvimento) muito correlacionado com target
* `VALOR_A_PAGAR` e `TAXA` têm pequena influência
* Variáveis de comportamento histórico se mostraram **essenciais**

### **4.3 Variáveis categóricas têm impacto forte**

WOE plots (`porte_woe.png`, `dominio_woe.png`, `reg_cep_woe.png`, `reg_ddd_woe.png`) revelaram:

* **Segmentos industriais específicos** possuem risco acima da média
* **Porte**: clientes pequenos apresentam maior risco
* **WOE de CEP e DDD** destacam regiões geográficas de alto risco
* **FLAG_PF**: clientes PF mostraram inadimplência superior

### **4.4 Variáveis financeiras**

* A renda do mês anterior é uma das features mais importantes no XGBoost
* Número de funcionários mostrou relação não linear com inadimplência

## **5. Feature Engineering**

As transformações aplicadas foram:

### **5.1 Variáveis temporais**

* Diferenças de datas
* Idade do cliente desde o cadastro
* Número de meses ativo

### **5.2 Históricos cumulativos por cliente**

Até o mês anterior:

* Número de cobranças anteriores
* Média de atrasos
* Máximo/mediana de atrasos
* % de pagamentos atrasados
* Tendência de atraso (últimos 3 meses)

(Destacadas nos shap plots `shap_xgb_fe_model.png`)

### **5.3 Transformações WoE para categóricas**

Aplicado especialmente para:

* SEGMENTO_INDUSTRIAL
* PORTE
* DOMINIO_EMAIL

Benefícios:

* Reduz cardinalidade
* Captura monotonicidade
* Facilita regressão logística


## **6. Modelagem**

Quatro modelos principais foram comparados:

| Modelo              | Feature Engineering | AUC       |
| ------------------- | ------------------- | --------- |
| Regressão Logística | Sem FE              | 0.733     |
| XGBoost             | Sem FE              | 0.838     |
| Regressão Logística | Com FE              | 0.827     |
| **XGBoost           | Com FE**            | **0.893** |

Gráfico salvo em:
`comparacao_modelos_modelling.png`


### **Principais conclusões da modelagem:**

1. **FE melhorou drasticamente a regressão logística**
2. **XGBoost superou os demais modelos com ampla margem**
3. As curvas ROC mostram clara dominância do XGBoost com FE
4. SHAP confirmou que:

   * histórico de atrasos
   * renda
   * porte
   * WOE de domínio
     são as features mais importantes.


## **8. Validação e Cuidados Contra Data Leakage**

Foram adotadas várias ações:

* Split temporal (train → validation → oot)
* Nenhuma estatística futura foi utilizada
* Todos os encoders foram treinados **somente com dados de desenvolvimento**
* Feature engineering replicado no teste sem usar informações proibidas


## **9. Modelo Final Escolhido**

### **XGBoost com Feature Engineering**

Justificativas:

* Maior AUC (0.893)
* Maior KS
* Feature importance coerente
* Robustez a outliers e não linearidades
* Compatível com produção

## **10. Geração da Submissão**

O pipeline final:

1. Carrega bases
2. Aplica os merges
3. Executa as mesmas transformações do treino
4. Aplica o modelo
5. Gera a coluna `PROBABILIDADE_INADIMPLENCIA`

Arquivo final:

`submissao_case.csv`


## **11. Estrutura do Repositório**

```
│
├── 1_data_understanding.ipynb
├── 2_eda.ipynb
├── 3_modelling.ipynb
│
├── src/
│   ├── data_utils.py
│   ├── eda_utils.py
│   ├── modelling_utils.py
│   └── submission_utils.py
│
├── assets/
│   ├── curva_roc_*.png
│   ├── shap_*.png
│   ├── *.png (WOE, correlações, distribuições)
│
├── requirements.txt
├── README.md  (este arquivo)
├── submissao_case.csv
└── main.py
```


### **Recomendações para o Time de Negócios**


**1. Utilizar o modelo como ferramenta de priorização de cobrança**

O modelo entrega **probabilidade de inadimplência**, permitindo priorizar ações de cobrança:

**Estratégia recomendada por faixas de risco**

| Faixa                          | Probabilidade                | Ação sugerida                                     |
| ------------------------------ | ---------------------------- | ------------------------------------------------- |
| **Baixo risco (0–0.20)**       | Clientes estáveis            | Enviar lembrete simples automatizado              |
| **Risco moderado (0.20–0.45)** | Clientes sensíveis           | Monitoramento ativo e comunicações personalizadas |
| **Alto risco (0.45–0.70)**     | Histórico instável           | Antecipar contato antes do vencimento             |
| **Risco crítico (>0.70)**      | Forte propensão a inadimplir | Priorizar cobrança manual / negociação            |

Essa priorização pode **reduzir custos operacionais**, aumentando eficiência sem ampliar carga de trabalho.

**2. Criar campanhas personalizadas por perfil de cliente**

Os SHAP values e análises WOE mostraram que variáveis como:

* **porte da empresa**
* **segmento industrial**
* **renda do mês anterior**

Assim, permitindo a criação **roteiros de cobrança, ofertas e flexibilizações específicas por perfil**, aumentando a assertividade das interações.

**3. Focar atenção em meses com maior sazonalidade de atraso**

A EDA revelou períodos do ano com **picos de inadimplência**, possivelmente ligados a:

* sazonalidade econômica,
* fluxo de caixa afetado em micro e pequenas empresas,
* datas específicas do setor.

