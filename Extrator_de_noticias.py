# imprime todas as noticias
import requests
from bs4 import BeautifulSoup
import csv

f = csv.writer(open('teste.csv', 'w'))
f.writerow(['Date', 'Title', 'Subtitle'])

def print_headline(headline_main):
    date = headline_main.find('small',{'class':'date'}).text
    title = headline_main.find('h2',{'class':'title2'}).text
    subtitle = headline_main.find('p',{'class':'gray3'}).text
    f.writerow([date,title,subtitle])

url = 'https://www.valor.com.br/empresas/cias'

for i in range(1,20):
    url = 'https://www.valor.com.br/empresas/cias?page=' + str(i) + '&page2='
    print(i)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = soup.find_all('div', {'class':'group'})
    
    for j in range (len(headlines)):
        headline_main = headlines[j]
        print_headline(headline_main)
        
