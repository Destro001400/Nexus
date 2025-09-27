from tokenizers import ByteLevelBPETokenizer
import os

# ----------------------------------------------------
# 0. ESCOLHA DO MODO DE TOKENIZAÇÃO
# ----------------------------------------------------
def choose_tokenizer_mode():
    """
    Solicita ao usuário para escolher o modo de tokenização (linguagem ou música)
    e retorna os caminhos correspondentes.
    """
    while True:
        print("--- Escolha o Modo de Tokenização ---")
        print("1. Linguagem (para treinar com data.txt)")
        print("2. Música (para treinar com datalyrics.txt)")
        choice = input("Digite 1 ou 2: ")

        base_dir = os.getcwd() # Usa o diretório atual

        if choice == '1':
            print("Modo de tokenização: Linguagem")
            text_file = os.path.join(base_dir, "data.txt")
            tokenizer_dir = os.path.join(base_dir, "nexus_tokenizer_lang")
            return text_file, tokenizer_dir
        elif choice == '2':
            print("Modo de tokenização: Música")
            text_file = os.path.join(base_dir, "datalyrics.txt")
            tokenizer_dir = os.path.join(base_dir, "nexus_tokenizer_music")
            return text_file, tokenizer_dir
        else:
            print("Opção inválida. Tente novamente.")

TEXT_FILE, TOKENIZER_DIR = choose_tokenizer_mode()

# Verifica se o arquivo de dados existe
if not os.path.exists(TEXT_FILE):
    print(f"ERRO: O arquivo de dados '{TEXT_FILE}' não foi encontrado.")
    print("Por favor, certifique-se de que o arquivo existe antes de criar o tokenizador.")
    exit()

# Cria o diretório do tokenizador se não existir
if not os.path.exists(TOKENIZER_DIR):
    os.makedirs(TOKENIZER_DIR)

# ----------------------------------------------------
# 1. INICIALIZAÇÃO E TREINAMENTO DO TOKENIZADOR
# ----------------------------------------------------

# Inicializa o tokenizador BPE
tokenizer = ByteLevelBPETokenizer(
    lowercase=True,
    add_prefix_space=True
)

# Treina o tokenizador
print(f"Treinando tokenizador com o arquivo: {TEXT_FILE}")
tokenizer.train(
    files=[TEXT_FILE],
    vocab_size=30000,  # Tamanho do vocabulário (ajustado para um valor comum)
    min_frequency=2,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ]
)
print("Treinamento do tokenizador concluído!")

# Salva o tokenizador
tokenizer.save_model(TOKENIZER_DIR)
print(f"Tokenizador salvo em: {TOKENIZER_DIR}")

# ----------------------------------------------------
# 2. TESTE RÁPIDO
# ----------------------------------------------------
print("\n--- Testando o tokenizador ---")
loaded_tokenizer = ByteLevelBPETokenizer.from_file(
    vocab_filename=os.path.join(TOKENIZER_DIR, "vocab.json"),
    merges_filename=os.path.join(TOKENIZER_DIR, "merges.txt")
)

test_text = "Este é um pequeno teste para verificar se o tokenizador funciona."
encoded = loaded_tokenizer.encode(test_text)

print(f"Texto original: \"{test_text}\"")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
print(f"Tamanho do vocabulário: {loaded_tokenizer.get_vocab_size()}")