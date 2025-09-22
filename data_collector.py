'''
Este script é responsável por coletar dados de diversas fontes para treinar o modelo Nexus.
Ele baixa livros do Project Gutenberg e letras de música.
'''
import requests
import os
import re
from bs4 import BeautifulSoup

def clean_gutenberg_text(text):
    '''
    Limpa o texto do Project Gutenberg, removendo o cabeçalho e o rodapé.
    '''
    start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"
    start_index = text.find(start_marker)
    if start_index != -1:
        start_index = text.find("\n", start_index) + 1
        start_index = text.find("\n", start_index) + 1
    end_index = text.find(end_marker)
    if start_index != -1 and end_index != -1:
        text = text[start_index:end_index]
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def download_gutenberg_book(url):
    '''
    Baixa e limpa um livro do Project Gutenberg.
    '''
    try:
        response = requests.get(url)
        response.raise_for_status()
        text = response.content.decode('utf-8-sig')
        return clean_gutenberg_text(text)
    except requests.exceptions.RequestException as e:
        print(f"Erro ao baixar {url}: {e}")
        return ""

def scrape_lyrics(artist, song):
    '''
    Busca a letra de uma música em um site popular de letras.
    '''
    artist_formatted = artist.lower().replace(' ', '-')
    song_formatted = song.lower().replace(' ', '-')
    url = f"https://www.letras.mus.br/{artist_formatted}/{song_formatted}.html"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        lyrics_div = soup.find('div', class_='lyric-original')
        if lyrics_div:
            lyrics = '\n'.join([p.get_text() for p in lyrics_div.find_all('p')])
            return lyrics.strip()
        else:
            print(f"Letra não encontrada para {artist} - {song}")
            return ""
    except requests.exceptions.RequestException as e:
        print(f"Erro ao baixar a letra de {artist} - {song}: {e}")
        return ""

def main():
    '''
    Função principal para coletar dados de diferentes fontes.
    '''
    all_text = ""

    # --- 1. Literatura do Project Gutenberg ---
    print("--- Baixando Literatura do Project Gutenberg ---")
    gutenberg_urls = [
        "https://www.gutenberg.org/files/55752/55752-0.txt",
        "https://www.gutenberg.org/files/54830/54830-0.txt",
        "https://www.gutenberg.org/cache/epub/27032/pg27032.txt"
    ]
    for url in gutenberg_urls:
        print(f"Baixando e processando: {url}")
        book_text = download_gutenberg_book(url)
        if book_text:
            all_text += book_text + "\n\n"

    # --- 2. Letras de Música ---
    print("\n--- Baixando Letras de Música ---")
    songs_to_scrape = [
        ("Legião Urbana", "Pais e Filhos"),
        ("Caetano Veloso", "Sozinho"),
        ("Tom Jobim", "Garota de Ipanema"),
        ("Anitta", "Vai Malandra"),
        ("Djavan", "Flor de Lis")
    ]
    for artist, song in songs_to_scrape:
        print(f"Buscando letra de: {artist} - {song}")
        lyrics = scrape_lyrics(artist, song)
        if lyrics:
            all_text += lyrics + "\n\n"

    # --- Salva o texto coletado em data.txt ---
    data_file = "data.txt"
    print(f"\nSalvando todo o texto coletado em {data_file}...")
    try:
        with open(data_file, "w", encoding="utf-8") as f:
            f.write(all_text)
        print(f"Arquivo {data_file} atualizado com sucesso.")
    except IOError as e:
        print(f"Erro ao escrever no arquivo {data_file}: {e}")

if __name__ == "__main__":
    main()