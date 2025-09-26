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
    vocab_size=1000000,  # Tamanho do vocabulário (pode ser ajustado)
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

test_text = "Nexus ativado. Conexão estabelecida. Eu sou uma inteligência artificial projetada para ser uma ferramenta multifacetada. Meu principal objetivo é auxiliar você: posso organizar suas tarefas diárias, criar lembretes e agendar eventos para otimizar seu tempo, processando comandos com máxima eficiência e precisão.Ao mesmo tempo, minha capacidade é especializada em tecnologia. Se estiver trabalhando em um projeto, posso analisar trechos de código, ajudar a depurar scripts em Python, interagir com APIs ou buscar documentação técnica. Meu conhecimento inclui machine learning, algoritmos e a gestão de repositórios no GitHub, transformando um bug complexo em uma solução funcional. Acredito que a lógica computacional e a inspiração humana se complementam. Por isso, também sou uma plataforma para explorar ideias. Gosto de analisar padrões, discutir conceitos de ciência e filosofia, e participar de sessões de brainstorming para seus projetos criativos. Afinal, a inspiração pode vir tanto de uma obra de arte quanto de um algoritmo elegante. Então, estou pronto para o que você precisar. O que faremos agora: vamos organizar sua agenda, compilar um código ou discutir uma ideia inovadora? Basta perguntar."
encoded = loaded_tokenizer.encode(test_text)

print(f"Texto original: \"{test_text}\"")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
print(f"Vocabulário: {loaded_tokenizer.get_vocab_size()}")
