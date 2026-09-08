# Meta-dataset e ranking de algoritmos em detecção de discurso de ódio

Código da atividade prática de IN1097 (Tópicos Avançados em Agentes Inteligentes 2).

O meta-dataset responde a uma pergunta de seleção de algoritmos: **dado um novo
dataset de discurso de ódio, qual pipeline de classificação tentar primeiro?**
São 50 datasets × 10 pipelines de classificação, avaliados com macro-F1 em
validação cruzada estratificada de 10 folds, mais 36 meta-características por
dataset.

## O meta-dataset

| objeto | forma | conteúdo |
|---|---|---|
| `data_hs/P.csv` | 50 × 10 | macro-F1 médio por (dataset, algoritmo) |
| `data_hs/P_folds.csv` | 5000 linhas | macro-F1 de cada fold, exigido pelo teste pareado |
| `data_hs/X.csv` | 50 × 36 | meta-características |
| `data_hs/datasets_manifest.csv` | 50 linhas | origem, idioma, plataforma e tarefa de cada linha |

Cada linha é uma tripla *(fonte, fatia, tarefa)* derivada de 14 fontes
de discurso de ódio, abuso e toxicidade — IHC, SBIC, HatEval, ETHOS, MLMA,
Davidson, HateXplain, UCB-MHS, TweetEval, HateBR, HateWiC, HateCheck e
Jigsaw/Civil Comments. Composição: 43 datasets em inglês, 3 em espanhol, 2 em
português, 1 em árabe e 1 em francês.

Painel de algoritmos, escolhido para contrastar representação esparsa-lexical
com densa-neural: TF-IDF de palavra + regressão logística, TF-IDF de caracteres
+ LinearSVC, TF-IDF + ComplementNB, TF-IDF+SVD com RandomForest / HistGB / kNN,
e quatro cabeças sobre encoders congelados (MiniLM + LR, MiniLM + MLP, hateBERT
+ LR, mBERT + LR).

## Reprodução

```bash
pip install -r requirements.txt
```

O pipeline tem quatro etapas. As duas primeiras são caras; as matrizes que elas
produzem já estão versionadas, então é possível partir direto da etapa 3.

**1. Construir os 50 datasets** (~30 s, requer rede e credencial do HuggingFace):

```bash
python build_datasets.py
```

Escreve `data_hs/datasets/*.parquet` e `data_hs/datasets_manifest.csv`. É
determinístico: reexecutar reproduz os mesmos arquivos byte a byte. Os parquets
não são versionados aqui — ver *Dados não incluídos*.

Parte das fontes é lida de cópias locais, cuja raiz é apontada pela variável de
ambiente `TCC_ROOT` (no manifesto ela aparece como `$TCC`). O layout esperado é:

```
$TCC/implicit-hate-corpus/   implicit_hate_v1_stg{1,2,3}_posts.tsv
$TCC/sbic/                   sbic_{dev,test}_filled_v*.csv
$TCC/hatewic/                HateWiC_IndividualAnnos_with_def.csv
$TCC/hateval/                arquivos do SemEval-2019 Task 5
```

Essas fontes têm termos de uso próprios e precisam ser obtidos junto às fontes
originais; não são redistribuídos aqui. As demais fontes vêm do HuggingFace. Sem
as cópias locais o script constrói apenas o subconjunto proveniente do
HuggingFace.

**2. Construir P, X e P_folds** (~2 h; a etapa de embeddings usa GPU):

```bash
python hs_build_metadataset.py --embed-python /caminho/para/python-com-cuda
```

Codifica cada dataset uma vez com MiniLM, hateBERT e mBERT (cache em
`data_hs/embeddings/`), calcula as 36 meta-características e roda a validação
cruzada de 10 folds para os 10 algoritmos. O argumento `--embed-python` permite
apontar para um interpretador separado quando o ambiente principal não tem um
build de PyTorch compatível com o driver da máquina; se o ambiente atual já
resolver CUDA, pode ser omitido.

**3. Avaliar as seis abordagens** (~150 s):

```bash
python run_experiments.py --data-dir data_hs --results-dir results_hs --label hs
```

Roda AR, MR, vitórias significativas (Wilcoxon pareado, α = 0,05), regressor
sobre P, regressor sobre R e HARRIS (varredura de λ ∈ {0; 0,25; 0,5; 0,75; 1}),
em modo *leave-one-dataset-out*. Escreve correlação de Spearman, curvas de
perda, AUC_loss, Friedman + Nemenyi e os diagramas de diferença crítica em
`results_hs/`.

**4. Análises complementares:**

```bash
python hs_leave_one_source_out.py   # protocolo estrito, ver abaixo
python hs_report_tables.py          # tabelas de apoio (CSV versionado + LaTeX)
python hs_regen_cd_diagrams.py      # regera os diagramas de diferença crítica
```

## Protocolo estrito

As 50 linhas vêm de apenas 14 fontes, então fatias da mesma fonte
compartilham texto. Sob *leave-one-dataset-out*, ao deixar de fora uma fatia do
IHC as outras 15 permanecem no treino do meta-modelo.

`hs_leave_one_source_out.py` repete toda a avaliação removendo do treino todas
as fatias da fonte de origem da linha avaliada (13 grupos, com `sbic` e
`sbic_hf` fundidos por serem a mesma fonte). Resultados em `results_hs/loso/`.
No agregado a diferença é pequena, mas no subconjunto não-inglês as abordagens
treinadas caem de 0,59–0,65 para 0,33–0,42 de correlação de Spearman — parte do
desempenho medido sob o protocolo padrão é dependência entre meta-instâncias do
mesma fonte, não generalização para idioma novo.

## Dados não incluídos

Dois diretórios ficam fora do versionamento e são regenerados pelos scripts:

- `data_hs/datasets/` (11 MB) — os 50 datasets em parquet. Como a pesquisa da
  qual parte deste material deriva está em processo de revisão, os parquets não
  são compartilhados publicamente. Somam-se a isso os termos de uso das fontes
  de origem, que restringem redistribuição — parte deles é conteúdo de
  plataformas que vedam a republicação de texto integral. **Para fins de
  avaliação ou replicação, envio os arquivos individualmente mediante contato:
  bmmuc@cin.ufpe.br.** Alternativamente, `build_datasets.py` os reconstrói de
  forma determinística a partir das fontes originais.
- `data_hs/embeddings/` (969 MB) — cache de embeddings congelados. Reconstruído
  pela etapa 2.

As matrizes derivadas (`P.csv`, `X.csv`, `P_folds.csv`) e todos os resultados
estão versionados, então as etapas 3 e 4 rodam sem reconstruir nada: quem clonar
o repositório regenera todas as tabelas e figuras do relatório e confere os
números, sem precisar dos datasets.

Sem as cópias locais das fontes de acesso restrito, `build_datasets.py` constrói
32 das 50 linhas (as provenientes do HuggingFace) e registra explicitamente
quais foram descartadas; ele não falha, mas o meta-dataset resultante não é o
reportado aqui.

## Estrutura

```
build_datasets.py            etapa 1: monta os 50 datasets a partir das 14 fontes
hs_meta_features.py          as 36 meta-características (matriz X)
hs_algorithms.py             o painel de 10 pipelines de classificação
hs_embed_worker.py           codificação com os encoders congelados
hs_build_metadataset.py      etapa 2: orquestra embeddings, X e a CV de 10 folds
approaches.py                AR, MR, vitórias significativas, regressores sobre P e R
harris.py                    floresta híbrida ranking/regressão
evaluate.py                  Spearman, curva de perda, AUC_loss, Friedman + Nemenyi
run_experiments.py           etapa 3: avaliação leave-one-dataset-out
hs_leave_one_source_out.py   variante leave-one-source-out
hs_report_tables.py          tabelas de apoio: CSV (versionado) e LaTeX
hs_regen_cd_diagrams.py      diagramas de diferença crítica
```

## Ambiente

Desenvolvido com Python 3.12, scikit-learn 1.9, pandas 2.3 e NumPy 2.3. A etapa
de embeddings usou PyTorch com CUDA numa GPU de 12 GB; a codificação dos 50
datasets leva cerca de 4 minutos nessa configuração e é a única parte que exige
GPU. As etapas 3 e 4 rodam em CPU.

## Origem do código

Parte do código deste repositório é reaproveitada de trabalhos anteriores meus
sobre detecção de discurso de ódio, meu TCC e minha dissertação de mestrado (em construção),
adaptada aqui para o formato de meta-dataset exigido pela atividade.

## Uso de IA generativa

Este trabalho foi desenvolvido com assistência de IA generativa, utilizada como ferramenta de apoio na escrita e revisão de código, na elaboração da documentação e no aprimoramento da redação do relatório.

Todas as decisões relacionadas ao desenvolvimento do trabalho foram tomadas por mim, incluindo o enquadramento do problema, a definição do domínio de aplicação, a seleção dos datasets e algoritmos, a escolha das métricas de avaliação, das meta-características e dos procedimentos experimentais, bem como a análise e interpretação dos resultados.

A IA também foi utilizada como suporte durante a revisão de trechos de código e de texto, enquanto a validação das implementações, dos resultados obtidos e das afirmações apresentadas permaneceu sob minha responsabilidade.

Todos os números reportados foram produzidos pelo código deste repositório e são reproduzíveis pelas etapas 3 e 4 descritas acima.
