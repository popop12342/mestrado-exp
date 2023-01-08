# Experimentos do Mestrado

Conjunto de experimentos para o trabalho de mestrado.

## Aumento por substituição por antônimo negado

O diretório `antonym_aug` tem as modificações no experimento do artigo EDA
para testar o aumento de dados textuais por substituição por antônimo negado.
Para rodar o experimento execute o notebook [`antonym_test.ipynb`](data/antonym_test.ipynb)

## GAN para data augmentation

O diretório `gan_aug` tem o experimento que utiliza um GAN para realizar
aumento de dados textuais. No subdiretório `gan_aug/optuna` experimos com ele
para otiimzação de hiper-parâmetros utilizando a ferramenta Optuna, rodando 
o arquivo princiapl gan_optuna.py.