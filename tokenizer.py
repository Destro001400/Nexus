from tokenizers import ByteLevelBPETokenizer
import os

# 1. Define o caminho para o arquivo de texto que criamos
TEXT_FILE = "data.txt"

# 2. Define o diretório onde o tokenizador será salvo
TOKENIZER_DIR = "nexus_tokenizer"
if not os.path.exists(TOKENIZER_DIR):
    os.makedirs(TOKENIZER_DIR)

# 3. Inicializa o tokenizador BPE
# ByteLevelBPETokenizer é uma boa escolha para LLMs, pois lida bem com caracteres desconhecidos.
tokenizer = ByteLevelBPETokenizer(
    lowercase=True,  # Converte tudo para minúsculas
    add_prefix_space=True # Adiciona um espaço no início da sequência
)

# 4. Treina o tokenizador
print(f"Treinando tokenizador com o arquivo: {TEXT_FILE}")
tokenizer.train(
    files=[TEXT_FILE],
    vocab_size=5000,  # Tamanho do vocabulário (pode ser ajustado)
    min_frequency=2,  # Tokens que aparecem menos de 2 vezes serão ignorados
    special_tokens=[
        "<s>",  # Token de início de sequência
        "<pad>", # Token de preenchimento
        "</s>", # Token de fim de sequência
        "<unk>", # Token para palavras desconhecidas
        "<mask>", # Token para mascaramento (usado em BERT, mas bom ter)
    ]
)
print("Treinamento concluído!")

# 5. Salva o tokenizador
tokenizer.save_model(TOKENIZER_DIR)
print(f"Tokenizador salvo em: {TOKENIZER_DIR}")

# 6. Teste rápido do tokenizador
print("\n--- Testando o tokenizador ---")
loaded_tokenizer = ByteLevelBPETokenizer.from_file(
    vocab_filename=os.path.join(TOKENIZER_DIR, "vocab.json"),
    merges_filename=os.path.join(TOKENIZER_DIR, "merges.txt")
)

test_text = "O Nexus é um modelo de IA que compõe música e pesquisa tecnologia."
encoded = loaded_tokenizer.encode(test_text)

print(f"Texto original: \"{test_text}\"")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
print(f"Vocabulário: {loaded_tokenizer.get_vocab_size()}")
