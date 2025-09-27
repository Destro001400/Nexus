"""
Este script é responsável por coletar dados de texto (livros e artigos) de diversas fontes 
para treinar o modelo Nexus de forma interativa.
"""
import requests
import os
import re
from bs4 import BeautifulSoup
import wikipediaapi
import gutenbergpy.textget
import random

def download_gutenberg_book(book_id):
    """
    Baixa e limpa um livro do Project Gutenberg usando a API.
    """
    try:
        # A função get_text_by_id retorna o texto bruto em bytes
        raw_text = gutenbergpy.textget.get_text_by_id(book_id)
        if raw_text:
            # A função strip_headers remove os cabeçalhos e rodapés do Gutenberg
            clean_text_bytes = gutenbergpy.textget.strip_headers(raw_text)
            # Decodifica para string, ignorando erros que podem ocorrer
            clean_text = clean_text_bytes.decode('utf-8', errors='ignore')
            # Remove espaços em branco extras para consistência
            clean_text = re.sub(r'\s+', ' ', clean_text)
            return clean_text.strip()
        else:
            print(f"Não foi possível encontrar o texto para o livro ID: {book_id}")
            return ""
    except Exception as e:
        print(f"Erro ao baixar o livro ID {book_id}: {e}")
        return ""

def search_gutenberg(query):
    """
    Busca livros no Project Gutenberg e retorna uma lista de (título, id).
    """
    print(f"Buscando livros sobre '{query}' no Project Gutenberg...")
    search_url = f"https://www.gutenberg.org/ebooks/search/?query={requests.utils.quote(query)}"
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        results = []
        for li in soup.find_all('li', class_='booklink'):
            book_id_tag = li.find('a', class_='link')
            title_tag = li.find('span', class_='title')
            if book_id_tag and title_tag:
                book_id_href = book_id_tag.get('href')
                book_id = book_id_href.split('/')[-1]
                title = title_tag.get_text(strip=True)
                if book_id.isdigit():
                    results.append((title, int(book_id)))
        
        if not results:
            print("Nenhum livro encontrado para esta busca.")
        return results

    except requests.exceptions.RequestException as e:
        print(f"Erro ao buscar no Project Gutenberg: {e}")
        return []

def get_gutenberg_books_interactive():
    """
    Permite ao usuário buscar e baixar livros do Project Gutenberg de forma interativa.
    """
    query = input("O que você gostaria de buscar no Project Gutenberg? ")
    search_results = search_gutenberg(query)
    
    if not search_results:
        return ""
        
    print("\nResultados da busca (mostrando os 10 primeiros):")
    for i, (title, book_id) in enumerate(search_results[:10]):
        print(f"{i+1}. {title} (ID: {book_id})")

    try:
        choices_str = input("\nDigite os números dos livros que você quer baixar, separados por vírgula (ex: 1,3,5): ")
        if not choices_str:
            return ""
        choices = [int(c.strip()) for c in choices_str.split(',')]
        
        all_books_text = ""
        for choice in choices:
            if 1 <= choice <= len(search_results[:10]):
                title, book_id = search_results[choice-1]
                print(f"Baixando '{title}' (ID: {book_id})...")
                book_text = download_gutenberg_book(book_id)
                if book_text:
                    all_books_text += book_text + "\n\n"
            else:
                print(f"Escolha inválida: {choice}")
        return all_books_text

    except ValueError:
        print("Entrada inválida. Por favor, use números separados por vírgula.")
        return ""

def get_wikipedia_articles(language, topic, count):
    """
    Busca artigos da Wikipedia relacionados a um tópico, pegando links da página principal.
    """
    print(f"--- Buscando {count} artigo(s) da Wikipedia em '{language}' sobre '{topic}' ---")
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent="Nexus Text Collector/1.0",
        language=language,
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )
    wiki_wiki._request_kwargs['timeout'] = 30
    
    main_page = wiki_wiki.page(topic)
    if not main_page.exists():
        print(f"Não foi possível encontrar uma página principal para '{topic}'. Tente um tópico mais específico.")
        return ""

    all_articles_text = ""
    # Adiciona o texto da página principal
    print(f"Buscando artigo principal: {topic}")
    all_articles_text += main_page.text + "\n\n"
    
    links = list(main_page.links.keys())
    
    if not links:
        print(f"A página '{topic}' não tem links para outros artigos.")
        return all_articles_text

    # -1 porque já pegamos o artigo principal
    count -= 1
    if count <= 0:
        return all_articles_text

    # Remove links que são de "ajuda", "portal", etc.
    links = [l for l in links if not any(x in l.lower() for x in ['ajuda:', 'portal:', 'special:', 'template:', 'file:'])]

    if count > len(links):
        print(f"Aviso: O número de artigos solicitados ({count+1}) é maior que o de links encontrados ({len(links)+1}). Buscando todos os links.")
        count = len(links)

    selected_titles = random.sample(links, count)

    for title in selected_titles:
        page = wiki_wiki.page(title)
        if page.exists():
            # Evitar páginas muito curtas (stubs) ou de desambiguação
            if len(page.text) > 500:
                print(f"Buscando artigo relacionado: {title}")
                all_articles_text += page.text + "\n\n"
            else:
                print(f"Artigo '{title}' muito curto ou página de desambiguação, pulando.")
        else:
            print(f"Artigo '{title}' não encontrado.")
            
    return all_articles_text

def main():
    """
    Função principal para coletar dados de diferentes fontes de forma interativa.
    """
    all_text = ""
    
    while True:
        print("\n--- Coletor de Dados Textuais Nexus ---")
        print("1. Baixar livros do Project Gutenberg")
        print("2. Baixar artigos da Wikipedia")
        print("3. Salvar dados coletados e sair")
        print("4. Sair sem salvar")
        
        choice = input("Escolha uma opção (1-4): ")
        
        if choice == '1':
            all_text += get_gutenberg_books_interactive()
        
        elif choice == '2':
            try:
                lang = input("Qual o idioma dos artigos (ex: 'pt' para português, 'en' para inglês)? ").lower()
                topic = input("Sobre qual tópico você quer os artigos? ")
                count = int(input(f"Quantos artigos sobre '{topic}' você quer baixar (incluindo o principal)? "))
                all_text += get_wikipedia_articles(lang, topic, count)
            except ValueError:
                print("Entrada inválida. Por favor, insira um número para a quantidade.")

        elif choice == '3':
            if not all_text:
                print("Nenhum texto foi coletado. Saindo.")
                break
            
            data_file = "data.txt"
            print(f"\nSalvando todo o texto coletado em {data_file}...")
            try:
                # 'a' para adicionar ao invés de sobrescrever
                with open(data_file, "a", encoding="utf-8") as f:
                    f.write(all_text)
                print(f"Dados adicionados com sucesso ao arquivo {data_file}.")
            except IOError as e:
                print(f"Erro ao escrever no arquivo {data_file}: {e}")
            break

        elif choice == '4':
            print("Saindo sem salvar.")
            break
            
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
