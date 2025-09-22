
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nexus_core.model import NexusModel
from nexus_core.dataset import NexusDataset, collate_fn

# Caminhos para o dataset e tokenizador
TEXT_FILE = "C:\\Users\\Casa\\Documents\\Nexus\\data.txt"
TOKENIZER_DIR = "C:\\Users\\Casa\\Documents\\Nexus\\nexus_tokenizer"
MODEL_SAVE_PATH = "C:\\Users\\Casa\\Documents\\Nexus\\Nexus_models\\nexus_model.pth" # Caminho para salvar o modelo

# -----------------------------------------------------
# 1. HIPERPARÂMETROS E CONFIGURAÇÕES DO MODELO
# -----------------------------------------------------
# Aqui definimos o tamanho e a forma do nosso modelo.
# Estes são valores de exemplo.

VOCAB_SIZE = 10000   # Tamanho do nosso dicionário (ex: 10.000 palavras únicas)
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
LR_SCHEDULER_PATIENCE = 2 # Paciência para o Learning Rate Scheduler (reduz LR se val_loss não melhorar por 2 épocas)
LR_SCHEDULER_FACTOR = 0.5 # Fator de redução do Learning Rate
EARLY_STOPPING_PATIENCE = 3 # Paciência para Early Stopping (para se val_loss não melhorar por 3 épocas)

# -----------------------------------------------------
# 2. FUNÇÕES DE SALVAR/CARREGAR MODELO
# -----------------------------------------------------
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Modelo salvo em {path}")

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Modelo carregado de {path}")
    return model

# -----------------------------------------------------
# 3. INICIALIZAÇÃO DOS COMPONENTES
# -----------------------------------------------------

# Define o dispositivo (GPU se disponível, senão CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicializa o Dataset e DataLoader
print("Inicializando o Dataset de Treino e Validação...")
train_dataset = NexusDataset(TEXT_FILE, TOKENIZER_DIR, SEQ_LEN, split='train', split_ratio=TRAIN_SPLIT_RATIO)
val_dataset = NexusDataset(TEXT_FILE, TOKENIZER_DIR, SEQ_LEN, split='val', split_ratio=TRAIN_SPLIT_RATIO)

VOCAB_SIZE = train_dataset.vocab_size # Atualiza VOCAB_SIZE dinamicamente com base no vocabulário do tokenizador
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"Dataset de Treino criado com {len(train_dataset)} sequências.")
print(f"Dataset de Validação criado com {len(val_dataset)} sequências.")
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

# (Item 26) Define a Função de Perda (Loss Function)
# CrossEntropyLoss é padrão para tarefas de classificação/geração de texto.
criterion = nn.CrossEntropyLoss()

# (Item 27) Define o Otimizador (Optimizer)
# AdamW é uma versão melhorada do Adam, muito popular hoje em dia.
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# (Item 28) O Learning Rate Scheduler
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min', # Monitora a perda de validação (min)
    factor=LR_SCHEDULER_FACTOR,
    patience=LR_SCHEDULER_PATIENCE,
    verbose=True
)

print(f"Modelo movido para o dispositivo: {device}")
print(f"Número de parâmetros: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# -----------------------------------------------------
# 4. LOOP DE TREINAMENTO
# -----------------------------------------------------

print("Iniciando loop de treinamento...")

best_val_loss = float('inf') # Inicializa com infinito para garantir que o primeiro modelo seja salvo
epochs_no_improve = 0 # Contador para Early Stopping

for epoch in range(N_EPOCHS):
    # --- Fase de Treinamento ---
    model.train() # Coloca o modelo em modo de treinamento
    total_train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataloader): # Usa train_dataloader
        inputs, targets = inputs.to(device), targets.to(device)

        # Cria uma máscara causal (look-ahead mask)
        tgt_mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool().to(device)

        optimizer.zero_grad()
        logits = model(inputs, tgt_mask)
        loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        loss.backward()

        # (Item 29) Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        optimizer.step()

        total_train_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Época [{epoch+1}/{N_EPOCHS}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss Treino: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Época [{epoch+1}/{N_EPOCHS}], Média da Loss de Treino: {avg_train_loss:.4f}")

    # --- Fase de Validação ---
    model.eval() # Coloca o modelo em modo de avaliação
    total_val_loss = 0
    with torch.no_grad(): # Desativa o cálculo de gradientes para validação
        for batch_idx, (inputs, targets) in enumerate(val_dataloader): # Usa val_dataloader
            inputs, targets = inputs.to(device), targets.to(device)

            tgt_mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool().to(device)

            logits = model(inputs, tgt_mask)
            loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"Época [{epoch+1}/{N_EPOCHS}], Média da Loss de Validação: {avg_val_loss:.4f}")

    # Step the learning rate scheduler
    lr_scheduler.step(avg_val_loss)

    # Salva o modelo se a loss de validação melhorar
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_model(model, MODEL_SAVE_PATH)
        print(f"Melhor modelo salvo com Loss de Validação: {best_val_loss:.4f}")
        epochs_no_improve = 0 # Reseta o contador de paciência
    else:
        epochs_no_improve += 1
        print(f"Validação Loss não melhorou por {epochs_no_improve} época(s).")

    # Early Stopping
    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping at epoch {epoch+1} as validation loss did not improve for {EARLY_STOPPING_PATIENCE} consecutive epochs.")
        break

print("Treinamento concluído!")



