# Experimentos do Mestrado

Conjunto de experimentos para o trabalho de mestrado.

## Aumento por substituição por antônimo negado

O diretório `antonym_aug` tem as modificações no experimento do artigo EDA
para testar o aumento de dados textuais por substituição por antônimo negado.
Para rodar o experimento execute o notebook [`antonym_test.ipynb`](data/antonym_test.ipynb)

## GAN para data augmentation

O diretório `gan_aug` tem o experimento que utiliza um GAN para realizar
aumento de dados textuais. No subdiretório `gan_aug/core` experimos com ele
para otimização de hiper-parâmetros utilizando a ferramenta Optuna, rodando
o arquivo gan_optuna.py.

Para rodar o experimento GAN-TEXTGEN + BERT execute o arquivo
`gan_textgeb_bert.py`. Para um conjunto diferente do subj faça

```bash
python gan_textgen_bert.py --dataset subj_005
```

### Data

Os datasets utilizados nesses experimentos deverão estar no diretório
`gan_aug/data`. Atualmente, suportamos os datasets os seguintes datasets:

| Dataset                                            | Tamanho de treinamento | Tamanho de validação | Tamanho total | Labels                         | Link |
| -------------------------------------------------- | ---------------------- | -------------------- | ------------- | ------------------------------ | ---- |
| SUBJ                                               | 9k                     | 1k                   | 10k           | *subjective*, *objetive*       | [Cornell](https://www.cs.cornell.edu/people/pabo/movie-review-data/) |
| AclIMDB                                            | 25k                    | 25k                  | 50k           | *negative*, *positive*         | [Kaggle](https://www.kaggle.com/datasets/pawankumargunjan/imdb-review) |
| Rotten400k                                         | 379k                   | 42k                  | 421k          | *negative*, *positive*         | [Kaggle](https://www.kaggle.com/datasets/talha002/rottentomatoes-400k-review) |
| Multilingual Task-Oriented Dialog Data (en/es/th)  | 30k/3,6k/2,1k          | 4k/2k/1,2k           | 43k/8k/5k     | *alarm*, *reminder*, *weather* | [Facebook](https://fb.me/multilingual_task_oriented_data) |
| Olist                                              | 33k                    | 4k                   | 47k           | *negative*, *positive*         | [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?resource=download&select=olist_order_reviews_dataset.csv) |
| Customer Support Emails - Ticket System - Helpdesk | 480                    | 120                  | 600            | *General Inquiry*, *Human Resources*, *Billing and Payments*, *Sales and Pre-Sales*, *IT Support*, *Customer Service*, *Product Support*, *Returns and Exchanges*, *Service Outages and Maintenance*, *Technical Support* | [Kaggle](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets/data) |
| Turkish Product Reviews                            | 211k                   | 24k                  | 235k           | *negative*, *positive*         | [Hugging Face](https://huggingface.co/datasets/fthbrmnby/turkish_product_reviews) |
| Multilingual Sentiments                            | 270k (all languages)   | 14k (all languages)  | 296k (all languages) | *positive*, *neural*, *negative* | [Hugging Face](https://huggingface.co/datasets/tyqiangz/multilingual-sentiments) |

### Aumento com LLM

É possível aumentar esses datasets com um LLM. Para isso, é preciso que tenha
disponível uma chave de API da OpenAI nas variáveis de ambiente com o nome
`OPENAI_API_KEY`, ou no arquivo .env. Então, será possível executar o aumento
de dados com LLM com o seguinte comando

```bash
python llm_augment.py --dataset subj_005
```

Ainda é possível utilizar outros modelos para o aumento pelos argumentos
`--model` e `--model_type` da seguinte forma:

```bash
python llm_augment.py --model meta-llama/Llama-3.2-3B-Instruct --model_type huggingface
```

Veja mais opções com

```bash
python llm_augment.py --help
```
