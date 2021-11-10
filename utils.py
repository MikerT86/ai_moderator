from bs4 import BeautifulSoup
import requests


def get_slang_words() -> list:

    resp = requests.get("http://www.netlingo.com/acronyms.php")
    soup = BeautifulSoup(resp.text, "html.parser")
    slangwards = []

    for div in soup.findAll('div', attrs={'class':'list_box3'}):
        for li in div.findAll('li'):
            for a in li.findAll('a'):
                slangwards.append(a.text.lower())

    return slangwards