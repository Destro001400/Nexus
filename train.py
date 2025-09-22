
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nexus_core.model import NexusModel
from nexus_core.dataset import NexusDataset, collate_fn

# Caminhos para o dataset e tokenizador
TEXT_FILE = "C:\\Users\\Casa\\Documents\\Nexus\\data.txt"
TOKENIZER_DIR = "C:\\Users\\Casa\\Documents\\Nexus\\nexus_tokenizer"


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

# -----------------------------------------------------
# 2. INICIALIZAÇÃO DOS COMPONENTES
# -----------------------------------------------------

# Define o dispositivo (GPU se disponível, senão CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicializa o Dataset e DataLoader
print("Inicializando o Dataset...")
dataset = NexusDataset(TEXT_FILE, TOKENIZER_DIR, SEQ_LEN)
VOCAB_SIZE = dataset.vocab_size # Atualiza VOCAB_SIZE dinamicamente com base no vocabulário do tokenizador
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
print(f"Dataset criado com {len(dataset)} sequências. Tamanho do vocabulário: {VOCAB_SIZE}")

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

# (Item 28) O Learning Rate Scheduler seria definido aqui, mas vamos manter simples por enquanto.

print(f"Modelo movido para o dispositivo: {device}")
print(f"Número de parâmetros: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# -----------------------------------------------------
# 3. LOOP DE TREINAMENTO
# -----------------------------------------------------

print("Iniciando loop de treinamento...")

# Coloca o modelo em modo de treinamento
model.train()

for epoch in range(N_EPOCHS):
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Cria uma máscara causal (look-ahead mask)
        # Garante que o modelo não veja tokens futuros
        tgt_mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool().to(device)

        # 1. Zera os gradientes da iteração anterior
        optimizer.zero_grad()

        # 2. Forward pass: Passa os dados pelo modelo
        logits = model(inputs, tgt_mask)

        # 3. Calcula a Loss
        # O Pytorch espera [Batch, Classes, SeqLen], então precisamos reformatar os tensores.
        loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))

        # 4. Backward pass: Calcula os gradientes da loss em relação aos parâmetros
        loss.backward()

        # (Item 29) O Gradient Clipping seria aplicado aqui, antes do optimizer.step()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 5. Atualiza os pesos do modelo
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Época [{epoch+1}/{N_EPOCHS}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Época [{epoch+1}/{N_EPOCHS}], Média da Loss: {avg_loss:.4f}")

print("Treinamento concluído!")


# -----------------------------------------------------
# 2. INICIALIZAÇÃO DOS COMPONENTES
# -----------------------------------------------------

# Define o dispositivo (GPU se disponível, senão CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# (Item 28) O Learning Rate Scheduler seria definido aqui, mas vamos manter simples por enquanto.

print(f"Modelo movido para o dispositivo: {device}")
print(f"Número de parâmetros: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# -----------------------------------------------------
# 3. SIMULAÇÃO DO LOOP DE TREINAMENTO
# -----------------------------------------------------
# Como ainda não temos um pipeline de dados, vamos criar dados falsos (placeholders)
# para simular o processo de treinamento.

print("\nIniciando loop de treinamento simulado...")

# Coloca o modelo em modo de treinamento
model.train()

for epoch in range(N_EPOCHS):
    # --- Em um treino real, aqui carregaríamos um lote de dados ---
    # Dados de entrada falsos (batch de sequências de números/tokens)
    dummy_input_seq = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
    # Dados de alvo falsos (a mesma sequência, deslocada em um para prever a próxima palavra)
    dummy_target_seq = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
    # Máscara de atenção para o decoder (para não ver o futuro)
    # Em um treino real, a máscara seria uma matriz triangular.
    dummy_mask = torch.ones(BATCH_SIZE, 1, SEQ_LEN, SEQ_LEN).bool().to(device)

    # 1. Zera os gradientes da iteração anterior
    optimizer.zero_grad()

    # 2. Forward pass: Passa os dados pelo modelo
    logits = model(dummy_input_seq, dummy_mask)

    # 3. Calcula a Loss
    # O Pytorch espera [Batch, Classes, SeqLen], então precisamos reformatar os tensores.
    loss = criterion(logits.view(-1, VOCAB_SIZE), dummy_target_seq.view(-1))

    # 4. Backward pass: Calcula os gradientes da loss em relação aos parâmetros
    loss.backward()

    # (Item 29) O Gradient Clipping seria aplicado aqui, antes do optimizer.step()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 5. Atualiza os pesos do modelo
    optimizer.step()

    print(f"Época [{epoch+1}/{N_EPOCHS}], Loss: {loss.item():.4f}")

print("\nTreinamento simulado concluído!")
