import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nexus_core.model import NexusModel
from nexus_core.dataset import NexusDataset, collate_fn
import os

# ----------------------------------------------------
# 0. ESCOLHA DO MODO DE TREINAMENTO
# ----------------------------------------------------
def choose_training_mode():
    """
    Solicita ao usuário para escolher o modo de treinamento (linguagem ou música)
    e retorna os caminhos correspondentes para os arquivos e diretórios.
    """
    while True:
        print("--- Escolha o Modo de Treinamento ---")
        print("1. Linguagem (para textos gerais, de data.txt)")
        print("2. Música (para letras de música, de datalyrics.txt)")
        choice = input("Digite 1 ou 2: ")

        base_dir = "C:\\Users\\Casa\\Documents\\Nexus"

        if choice == '1':
            print("Modo de treinamento: Linguagem")
            text_file = os.path.join(base_dir, "data.txt")
            tokenizer_dir = os.path.join(base_dir, "nexus_tokenizer_lang")
            model_save_path = os.path.join(base_dir, "Nexus_models", "nexus_model_lang.pth")
            return text_file, tokenizer_dir, model_save_path
        elif choice == '2':
            print("Modo de treinamento: Música")
            text_file = os.path.join(base_dir, "datalyrics.txt")
            tokenizer_dir = os.path.join(base_dir, "nexus_tokenizer_music")
            model_save_path = os.path.join(base_dir, "Nexus_models", "nexus_model_music.pth")
            return text_file, tokenizer_dir, model_save_path
        else:
            print("Opção inválida. Tente novamente.")

TEXT_FILE, TOKENIZER_DIR, MODEL_SAVE_PATH = choose_training_mode()

# Verifica se os arquivos e diretórios necessários existem
if not os.path.exists(TEXT_FILE):
    print(f"ERRO: O arquivo de dados '{TEXT_FILE}' não foi encontrado.")
    print("Por favor, execute o script de coleta de dados apropriado primeiro.")
    exit()

if not os.path.exists(TOKENIZER_DIR):
    print(f"ERRO: O diretório do tokenizador '{TOKENIZER_DIR}' não foi encontrado.")
    print(f"Por favor, execute o script 'tokenizer.py' no arquivo '{os.path.basename(TEXT_FILE)}' para criar o tokenizador antes de treinar.")
    exit()

# ----------------------------------------------------
# 1. HIPERPARÂMETROS E CONFIGURAÇÕES DO MODELO
# ----------------------------------------------------
# Aqui definimos o tamanho e a forma do nosso modelo.
# Estes são valores de exemplo.

VOCAB_SIZE = 10000   # Tamanho do nosso dicionário (será atualizado dinamicamente)
D_MODEL = 512      # Dimensão principal do modelo (tamanho dos embeddings)
N_LAYERS = 6       # Número de DecoderBlocks empilhados
N_HEADS = 8        # Número de cabeças de atenção
D_K = D_V = D_MODEL // N_HEADS # Dimensão das queries, keys e values
D_HID = 2048       # Dimensão da camada oculta na rede Feed-Forward
DROPOUT = 0.1

# Configurações de Treinamento
LEARNING_RATE = 1e-4
N_EPOCHS = 10
BATCH_SIZE = 32
SEQ_LEN = 128
TRAIN_SPLIT_RATIO = 0.9 # 90% para treino, 10% para validação

# Novas configurações para otimização
MAX_GRAD_NORM = 1.0 # Limite para o Gradient Clipping
LR_SCHEDULER_PATIENCE = 2 # Paciência para o Learning Rate Scheduler
LR_SCHEDULER_FACTOR = 0.5 # Fator de redução do Learning Rate
EARLY_STOPPING_PATIENCE = 3 # Paciência para Early Stopping

# ----------------------------------------------------
# 2. FUNÇÕES DE SALVAR/CARREGAR MODELO
# ----------------------------------------------------
def save_model(model, path):
    # Garante que o diretório de salvamento exista
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Modelo salvo em {path}")

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Modelo carregado de {path}")
    return model

# ----------------------------------------------------
# 3. INICIALIZAÇÃO DOS COMPONENTES
# ----------------------------------------------------

# Define o dispositivo (GPU se disponível, senão CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Inicializa o Dataset e DataLoader
print("Inicializando Datasets e DataLoaders...")
train_dataset = NexusDataset(TEXT_FILE, TOKENIZER_DIR, SEQ_LEN, split='train', split_ratio=TRAIN_SPLIT_RATIO)
val_dataset = NexusDataset(TEXT_FILE, TOKENIZER_DIR, SEQ_LEN, split='val', split_ratio=TRAIN_SPLIT_RATIO)

VOCAB_SIZE = train_dataset.vocab_size # Atualiza VOCAB_SIZE dinamicamente
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"Dataset de Treino: {len(train_dataset)} sequências.")
print(f"Dataset de Validação: {len(val_dataset)} sequências.")
print(f"Tamanho do vocabulário: {VOCAB_SIZE}")

# Inicializa o nosso modelo Nexus
print("Inicializando o modelo Nexus...")
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

# Define a Função de Perda (Loss Function)
criterion = nn.CrossEntropyLoss()

# Define o Otimizador (Optimizer)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# O Learning Rate Scheduler
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=LR_SCHEDULER_FACTOR,
    patience=LR_SCHEDULER_PATIENCE
)

print(f"Número de parâmetros: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ----------------------------------------------------
# 4. LOOP DE TREINAMENTO
# ----------------------------------------------------

print("Iniciando loop de treinamento...")

best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(N_EPOCHS):
    # --- Fase de Treinamento ---
    model.train()
    total_train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        tgt_mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool().to(device)

        optimizer.zero_grad()
        logits = model(inputs, tgt_mask)
        loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        total_train_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Época [{epoch+1}/{N_EPOCHS}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss Treino: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Época [{epoch+1}/{N_EPOCHS}], Média da Loss de Treino: {avg_train_loss:.4f}")

    # --- Fase de Validação ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            tgt_mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool().to(device)
            logits = model(inputs, tgt_mask)
            loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"Época [{epoch+1}/{N_EPOCHS}], Média da Loss de Validação: {avg_val_loss:.4f}")

    lr_scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_model(model, MODEL_SAVE_PATH)
        print(f"Melhor modelo salvo com Loss de Validação: {best_val_loss:.4f}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"Validação Loss não melhorou por {epochs_no_improve} época(s).")

    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping at epoch {epoch+1} as validation loss did not improve for {EARLY_STOPPING_PATIENCE} consecutive epochs.")
        break

print("Treinamento concluído!")