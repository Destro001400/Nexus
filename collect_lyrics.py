
import lyricsgenius
import os

# -------------------------------------------------------------------------
# --- CONFIGURAÇÕES - PREENCHA ANTES DE EXECUTAR ---
# -------------------------------------------------------------------------

# 1. Cole seu Client Access Token do Genius.com aqui
GENIUS_ACCESS_TOKEN = "dga5f4cN5jNpm8RtBjbo3i8l48SKxCMqK2RNMRdQl0ap9D5CeeZnNrrK8fV2hcqL"

# 2. Defina os artistas cujas músicas você quer baixar
#    Pode ser um ou mais, separados por vírgula.
#    Exemplo: ARTISTAS = ["Djavan", "Legião Urbana"]
ARTISTAS = ["Djavan", "Legião Urbana", "Caetano Veloso", "Anitta", "Tom Jobim"]

# 3. Defina o número máximo de músicas por artista
MAX_SONGS_PER_ARTIST = 10

# 4. Defina o nome do arquivo onde as letras serão salvas
OUTPUT_FILENAME = "data.txt"

# -------------------------------------------------------------------------
# --- CÓDIGO DE COLETA - NÃO PRECISA ALTERAR ---
# -------------------------------------------------------------------------

def collect_lyrics():
    """
    Coleta letras de músicas dos artistas especificados usando a API do Genius
    e as salva em um único arquivo de texto.
    """
    if GENIUS_ACCESS_TOKEN == "SEU_TOKEN_DE_ACESSO_AQUI":
        print("ERRO: Por favor, insira seu Client Access Token do Genius na variável 'GENIUS_ACCESS_TOKEN'.")
        return

    if "NOME_DO_ARTISTA_AQUI" in ARTISTAS:
        print("ERRO: Por favor, defina os artistas que você deseja coletar na lista 'ARTISTAS'.")
        return

    print(f"Iniciando a coleta de letras para os artistas: {', '.join(ARTISTAS)}")
    print(f"Máximo de {MAX_SONGS_PER_ARTIST} músicas por artista.")
    print("-" * 30)

    genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, verbose=False)
    all_lyrics = []

    for artist_name in ARTISTAS:
        try:
            print(f"Buscando artista: {artist_name}...")
            artist = genius.search_artist(artist_name, max_songs=MAX_SONGS_PER_ARTIST, sort="popularity")

            if artist:
                print(f"Encontrado! Coletando letras de {len(artist.songs)} músicas...")
                for song in artist.songs:
                    if song and song.lyrics:
                        # Remove o cabeçalho "EmbedShare URLCopyEmbedCopy" e outras partes indesejadas
                        cleaned_lyrics = song.lyrics.split("Lyrics", 1)[-1]
                        if "You might also like" in cleaned_lyrics:
                            cleaned_lyrics = cleaned_lyrics.split("You might also like")[0]
                        
                        # Remove os números no final das linhas (ex: 123Embed)
                        cleaned_lyrics = "\n".join(line.rstrip('0123456789') for line in cleaned_lyrics.split("\n"))
                        
                        all_lyrics.append(cleaned_lyrics.strip())
            else:
                print(f"Artista '{artist_name}' não encontrado.")
            print("-" * 30)

        except Exception as e:
            print(f"Ocorreu um erro ao processar o artista {artist_name}: {e}")
            print("-" * 30)

    if all_lyrics:
        print(f"Coleta finalizada. Total de {len(all_lyrics)} letras obtidas.")
        
        # Salva as letras no arquivo, substituindo o conteúdo existente
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            for lyrics in all_lyrics:
                f.write(lyrics + "\n\n---\n\n") # Separador entre as letras
        
        print(f"Todas as letras foram salvas em '{OUTPUT_FILENAME}'.")
    else:
        print("Nenhuma letra foi coletada.")

if __name__ == "__main__":
    collect_lyrics()
