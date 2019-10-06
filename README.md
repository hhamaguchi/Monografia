Projeto Monografia
Desenvolvedor: Henrique Hamaguchi 
Data da criação do documento: 06/10/2019 
Data da última revisão do documento: 06/10/2019
1.	Projeto: Estudo do impacto de notícias sobre o preço das ações
2.	Para este estudo é necessário desenvolvimento de um extrator de notícias do site Valor Econômico. E para analisar as notícias são utilizadas três ferramentas de mineração textual: Support Vector Machine (SVM), Random Forest e Naive Bayes, estas notícias serão classificadas em notícias que tem impacto positivo ou negativo sobre a variação dos preços dos ativos.
3.	Requisitos do projeto:
  •	Excel Microsoft versão 1907
  •	Spyder (Pyhton 3.7)
  •	Bibliotecas utilizadas em Pyhton:
    a.	BeautifulSoup: utilizada para extrair informações de websites;
    b.	CSV: biblioteca utilizada para facilitar a leitura e gravação de dados em arquivos CSV;
    c.	NLTK: é um conjunto de bibliotecas com funcionalidades de análise de texto tais como classificação, tokenização, stemmização e análise semântica;
    d.	NumPy: fornece funções básicas de álgebra linear, matrizes esparsas e de otimização;
    e.	Pandas: fornece estruturas de dados de alto nível e várias ferramentas de análise. A ideia consiste em resolver problemas complexos facilmente, com poucos comandos. O Pandas tem muitas funcionalidades de preparação de dados como agrupar, combinar e filtrar dados, além de trabalhar com séries temporais também;
    f.	Requests: útilizada para acesso à web;
    g.	Scikit-Learn: fornece ferramentas para machine learning e modelagem estatística, tais como regressão, clusterização, redução de dimensionalidade e classificação.
4.	Como rodar os programas:
  •	Extrator de notícias do site Valor Econômico: carregar o código Extrator_de_noticias.py. 
    Parâmetros:
    a.	Arquivo de saída (linha 14)
    b.	Determinar quantas páginas são copiadas as informações de título, subtítulo e data de publicação da notícia (linha 37)
    c.	Principal: através dos classificadores tenta-se acertar a tendência do mercado
  •	Utilização de classificadores para determinar a tendencia do preço das ações: Principal.py 
    Parâmetros:
    a.	Determinar o tipo de análise: (1) título, (1) subtítulo e (3) título +subtítulo (linha 50)
    b.	Determinar a janela de dados que será composta por dados de treinamento mais dados de teste (linha 52)
    c.	Determinar o uso ou não da função de stemming (linha 65-70)
    d.	Determinar o percentual de dados da janela que será dados de treinamento e dados de teste (linha 76)
    5.	Algoritmos de classificação utilizados: Naive Bayes Multinomial, Support Vector Machine (lienar) e Random Forest. 
