'''
Este script fornece uma interface de linha de comando interativa para filtrar
e limpar arquivos de texto, permitindo que o usuário aplique vários filtros
em sequência e salve o resultado.
'''
import os
import re
import textwrap

# --- Funções de Filtro ---

def is_portuguese(text):
    """
    Verifica se um texto é provavelmente em português com base em uma lista de stop words.
    """
    portuguese_words = [
        'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'à', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'há', 'nos', 'já', 'está', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'era', 'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'estão', 'você', 'tinha', 'foram', 'essa', 'num', 'nem', 'suas', 'meu', 'às', 'minha', 'numa', 'pelos', 'elas', 'havia', 'seja', 'qual', 'será', 'nós', 'tenho', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'fosse', 'dele', 'tu', 'te', 'vocês', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo', 'estou', 'está', 'estamos', 'estão', 'estive', 'esteve', 'estivemos', 'estiveram', 'estava', 'estávamos', 'estavam', 'estivera', 'estivéramos', 'esteja', 'estejamos', 'estejam', 'estivesse', 'estivéssemos', 'estivessem', 'estiver', 'estivermos', 'estiverem', 'hei', 'há', 'havemos', 'hão', 'houve', 'houvemos', 'houveram', 'houvera', 'houvéramos', 'haja', 'hajamos', 'hajam', 'houvesse', 'houvéssemos', 'houvessem', 'houver', 'houvermos', 'houverem', 'houverei', 'houverá', 'houveremos', 'houverão', 'houveria', 'houveríamos', 'houveriam', 'sou', 'é', 'somos', 'são', 'fui', 'foi', 'fomos', 'foram', 'era', 'éramos', 'eram', 'fora', 'fôramos', 'seja', 'sejamos', 'sejam', 'fosse', 'fôssemos', 'fossem', 'for', 'formos', 'forem', 'serei', 'será', 'seremos', 'serão', 'seria', 'seríamos', 'teriam'
    ]
    words = re.findall(r'\w+', text.lower())
    if not words:
        return False
    portuguese_word_count = sum(1 for word in words if word in portuguese_words)
    return (portuguese_word_count / len(words)) > 0.4

def remove_duplicates(lines):
    """Remove linhas duplicadas de uma lista de linhas."""
    return list(dict.fromkeys(lines))

def clean_gutenberg(lines):
    """Remove cabeçalhos e rodapés de livros do Project Gutenberg."""
    text = "\n".join(lines)
    # Tenta encontrar os marcadores de início e fim
    start_marker = re.search(r'\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK .* \*\*\*', text)
    end_marker = re.search(r'\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK .* \*\*\*', text)
    
    if start_marker and end_marker:
        # Extrai o texto entre os marcadores
        text = text[start_marker.end():end_marker.start()]
        return text.strip().split('\n')
    
    # Se não encontrar marcadores, retorna as linhas originais para não perder dados
    return lines

def sentence_per_line(lines):
    """Formata o texto para ter uma sentença por linha."""
    text = " ".join(lines)
    # Substitui múltiplos espaços/quebras de linha por um único espaço
    text = re.sub(r'\s+', ' ', text)
    # Regex para separar sentenças, mantendo os delimitadores (., !, ?)
    # Esta regex tenta evitar quebras em abreviações como "Sr." ou "p. ex."
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?])\s', text)
    # Remove espaços extras e garante que cada sentença esteja em sua linha
    return [s.strip() for s in sentences if s.strip()]

# --- Funções de Interface ---

def display_menu(title, options):
    """Exibe um menu formatado."""
    print(f"\n--- {title} ---")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    print("-" * (len(title) + 8))

def select_file(directory, extension):
    """Lista arquivos com uma extensão e permite ao usuário escolher um."""
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    if not files:
        print(f"Nenhum arquivo '{extension}' encontrado em '{directory}'.")
        return None
    
    display_menu(f"Selecione um arquivo '{extension}' para carregar", files)
    try:
        choice = int(input(f"Escolha um arquivo (1-{len(files)}): "))
        if 1 <= choice <= len(files):
            return os.path.join(directory, files[choice - 1])
        else:
            print("Opção inválida.")
            return None
    except ValueError:
        print("Entrada inválida. Por favor, insira um número.")
        return None

# --- Função Principal ---

def main():
    """Função principal que executa o loop do menu interativo."""
    
    current_content_lines = []
    original_line_count = 0
    file_loaded = ""

    main_menu_options = [
        "Carregar arquivo de texto (.txt)",
        "Aplicar filtros",
        "Ver estatísticas",
        "Salvar arquivo filtrado",
        "Sair"
    ]
    
    filter_menu_options = [
        "Remover linhas duplicadas",
        "Remover cabeçalhos/rodapés do Gutenberg",
        "Manter apenas texto em português",
        "Filtrar por comprimento da linha",
        "Formatar quebras de linha (1 sentença por linha)",
        "Remover linhas com padrão (Regex)",
        "Manter apenas linhas com padrão (Regex)",
        "Voltar ao menu principal"
    ]

    while True:
        display_menu("Filtro de Dados Interativo", main_menu_options)
        main_choice = input("Escolha uma opção: ")

        if main_choice == '1':
            # Carregar arquivo
            filepath = select_file('.', '.txt')
            if filepath:
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                        current_content_lines = f.read().splitlines()
                    original_line_count = len(current_content_lines)
                    file_loaded = filepath
                    print(f"Arquivo '{file_loaded}' carregado com {original_line_count} linhas.")
                except Exception as e:
                    print(f"Erro ao carregar o arquivo: {e}")
        
        elif main_choice == '2':
            # Aplicar filtros
            if not file_loaded:
                print("Nenhum arquivo carregado. Por favor, carregue um arquivo primeiro (Opção 1).")
                continue

            while True:
                display_menu("Menu de Filtros", filter_menu_options)
                filter_choice = input("Escolha um filtro para aplicar: ")
                
                lines_before = len(current_content_lines)

                if filter_choice == '1':
                    current_content_lines = remove_duplicates(current_content_lines)
                    print("Filtro de linhas duplicadas aplicado.")
                elif filter_choice == '2':
                    current_content_lines = clean_gutenberg(current_content_lines)
                    print("Filtro de limpeza do Gutenberg aplicado.")
                elif filter_choice == '3':
                    current_content_lines = [line for line in current_content_lines if is_portuguese(line)]
                    print("Filtro de idioma português aplicado.")
                elif filter_choice == '4':
                    try:
                        min_len = int(input("Digite o comprimento mínimo da linha: "))
                        max_len = int(input("Digite o comprimento máximo da linha: "))
                        current_content_lines = [line for line in current_content_lines if min_len <= len(line) <= max_len]
                        print("Filtro de comprimento de linha aplicado.")
                    except ValueError:
                        print("Entrada inválida. Use números inteiros.")
                elif filter_choice == '5':
                    current_content_lines = sentence_per_line(current_content_lines)
                    print("Filtro de uma sentença por linha aplicado.")
                elif filter_choice == '6':
                    pattern = input("Digite o padrão Regex para REMOVER as linhas: ")
                    try:
                        current_content_lines = [line for line in current_content_lines if not re.search(pattern, line)]
                        print("Filtro de remoção por Regex aplicado.")
                    except re.error as e:
                        print(f"Erro no padrão Regex: {e}")
                elif filter_choice == '7':
                    pattern = input("Digite o padrão Regex para MANTER as linhas: ")
                    try:
                        current_content_lines = [line for line in current_content_lines if re.search(pattern, line)]
                        print("Filtro de manutenção por Regex aplicado.")
                    except re.error as e:
                        print(f"Erro no padrão Regex: {e}")
                elif filter_choice == '8':
                    break
                else:
                    print("Opção inválida.")

                lines_after = len(current_content_lines)
                print(f"Linhas afetadas nesta etapa: {lines_before - lines_after}")
                print(f"Total de linhas atual: {lines_after}")

        elif main_choice == '3':
            # Ver estatísticas
            if not file_loaded:
                print("Nenhum arquivo carregado.")
            else:
                print("\n--- Estatísticas ---")
                print(f"Arquivo original: {file_loaded}")
                print(f"Linhas originais: {original_line_count}")
                print(f"Linhas atuais:    {len(current_content_lines)}")
                print(f"Linhas removidas: {original_line_count - len(current_content_lines)}")
                print("-" * 20)

        elif main_choice == '4':
            # Salvar arquivo
            if not file_loaded:
                print("Nenhum arquivo carregado para salvar.")
                continue
            
            output_filename = input("Digite o nome do arquivo de saída (ex: 'dados_filtrados.txt'): ")
            if not output_filename.endswith('.txt'):
                output_filename += '.txt'
            
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    f.write("\n".join(current_content_lines))
                print(f"Arquivo salvo com sucesso como '{output_filename}'.")
            except Exception as e:
                print(f"Erro ao salvar o arquivo: {e}")

        elif main_choice == '5':
            print("Saindo do script.")
            break
        
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()