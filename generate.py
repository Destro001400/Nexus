
import torch
import torch.nn as nn
from tokenizers import ByteLevelBPETokenizer
import os

from nexus_core.model import NexusModel

# Caminhos para o tokenizador e modelo salvo
TOKENIZER_DIR = "C:\\Users\\Casa\\Documents\\Nexus\\nexus_tokenizer"
MODEL_SAVE_PATH = "C:\\Users\\Casa\\Documents\\Nexus\\Nexus_models\\nexus_model.pth"

# Hiperparâmetros do modelo (devem ser os mesmos usados no treinamento)
# Estes são valores de exemplo, ajuste conforme o seu train.py
VOCAB_SIZE = 357 # Atualizado com base no seu treinamento
D_MODEL = 512
N_LAYERS = 6
N_HEADS = 8
D_K = D_V = D_MODEL // N_HEADS
D_HID = 2048
DROPOUT = 0.1
SEQ_LEN = 128 # Comprimento da sequência usado no treinamento

def load_tokenizer(tokenizer_dir):
    tokenizer = ByteLevelBPETokenizer.from_file(
        vocab_filename=os.path.join(tokenizer_dir, "vocab.json"),
        merges_filename=os.path.join(tokenizer_dir, "merges.txt")
    )
    return tokenizer

def load_nexus_model(model_path, device):
    model = NexusModel(
        n_token=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layer=N_LAYERS,
        n_head=N_HEADS,
        d_k=D_K,
        d_v=D_V,
        d_hid=D_HID,
        dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Coloca o modelo em modo de avaliação
    print(f"Modelo Nexus carregado de {model_path}")
    return model

def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, device='cpu'):
    model.eval()
    encoded_prompt = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(encoded_prompt, dtype=torch.long).unsqueeze(0).to(device) # Adiciona dimensão do batch

    generated_ids = input_ids

    with torch.no_grad():
        for _ in range(max_length):
            # Se a sequência de entrada for maior que SEQ_LEN, pegamos apenas os últimos SEQ_LEN tokens
            current_input = generated_ids[:, -SEQ_LEN:] if generated_ids.shape[1] > SEQ_LEN else generated_ids

            # Cria uma máscara causal (look-ahead mask)
            tgt_mask = torch.triu(torch.ones(current_input.shape[1], current_input.shape[1]), diagonal=1).bool().to(device)

            logits = model(current_input, tgt_mask)
            # Pega os logits do último token gerado
            next_token_logits = logits[:, -1, :] / temperature
            
            # Aplica softmax para obter probabilidades e amostra o próximo token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            # Decodifica o token gerado para verificar se é um token de fim de sequência (se você tiver um)
            # Por enquanto, vamos apenas verificar o comprimento máximo

    return tokenizer.decode(generated_ids.squeeze(0).tolist())

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Carregando tokenizador...")
    tokenizer = load_tokenizer(TOKENIZER_DIR)

    print("Carregando modelo Nexus...")
    model = load_nexus_model(MODEL_SAVE_PATH, device)

    print("\n--- Gerador de Texto Nexus ---")
    print("Digite um prompt para o modelo gerar texto. Digite 'sair' para encerrar.")

    while True:
        user_prompt = input("Seu prompt: ")
        if user_prompt.lower() == 'sair':
            break
        
        print("Gerando texto...")
        generated_text = generate_text(model, tokenizer, user_prompt, max_length=100, temperature=0.7, device=device)
        print("\nTexto Gerado:")
        print(generated_text)
        print("-" * 30)
