
Sistema de reconhecimento de números feito com python, utilizando redes neurais. // assim espero 
é uma rede neural bem simples mas...
A neural trabalhará com 25 entradas, 15 camadas ocultas e 10 saídas.

No Codigo estou ultilizando uma base de dados propria, a taxa de aprendizagem é 0.03 
voce pode fazer teste como diminuir ou almentar a taxa de aprnedizagem, como pode mexer tambem na quantidade de testes que esta em 1000 no momento e possivel alterar a iteraçao
no momento ela esta em 100, a cada 100 vezes eles testa os resultados 
Os números
Os números serão como a que esta junto ai, composto por uma matriz 5 x 5, onde o valor de cada pixel servirá para que a rede neural seja treinada e reconheça o número.
Os Pixels assumirão valores 0 (Para os pixels brancos no caso nao pintados) e 1 (Para os pretos os pintados), formando um vetor que representará o número.

O Resultado
O resultado, será representado por um vetor com valor 1 na posição no número correspondente.


o codigo esta comentado.

um exemplo de aprendizagem e trocar a quandtidade de treinos para 200 e "array = neuralNetwork.activate" para outro dos numeros, provavelmemnte a rede nao 
encontrara um numero na primeira tenativa, tendo que ser feita algumas vezs ate ela chegar no numero certo.