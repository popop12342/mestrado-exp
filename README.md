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

Os datasets utilizados nesses experimentos deverão estar no diretório
`gan_aug/data`. Atualmente, suportamos os datasets subj e
[acllmbd](http://www.aclweb.org/anthology/P11-1015%7D) que foi baixado do
[Kaggle](https://www.kaggle.com/datasets/pawankumargunjan/imdb-review).
