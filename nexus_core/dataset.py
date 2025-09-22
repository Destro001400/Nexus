import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
import os

class NexusDataset(Dataset):
    """
    Um Dataset customizado para o Nexus, que carrega texto, tokeniza e prepara
    sequências para o treinamento de um modelo de linguagem.
    """
    def __init__(self, text_file, tokenizer_dir, seq_len, split='train', split_ratio=0.9):
        """
        Inicializa o Dataset.

        Args:
            text_file (str): Caminho para o arquivo de texto bruto.
            tokenizer_dir (str): Caminho para o diretório onde o tokenizador está salvo.
            seq_len (int): O comprimento máximo das sequências de entrada/saída.
            split (str): 'train' ou 'val' para indicar qual parte do dataset usar.
            split_ratio (float): A proporção para o conjunto de treinamento (e.g., 0.9 para 90% treino, 10% val).
        """
        super().__init__()
        self.seq_len = seq_len
        self.split = split
        self.split_ratio = split_ratio

        # 1. Carrega o tokenizador treinado
        self.tokenizer = ByteLevelBPETokenizer.from_file(
            vocab_filename=os.path.join(tokenizer_dir, "vocab.json"),
            merges_filename=os.path.join(tokenizer_dir, "merges.txt")
        )
        self.vocab_size = self.tokenizer.get_vocab_size()

        # 2. Lê o texto completo do arquivo
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()

        # 3. Tokeniza o texto completo
        encoded_text = self.tokenizer.encode(text)
        all_token_ids = encoded_text.ids

        # Calcula o número total de sequências que podemos extrair do texto completo
        total_possible_sequences = len(all_token_ids) - seq_len

        if total_possible_sequences <= 0:
            raise ValueError(f"O texto é muito curto para o seq_len={seq_len}. "
                             f"Tamanho do texto tokenizado: {len(all_token_ids)}")

        # Divide os dados em treino e validação
        split_idx = int(total_possible_sequences * split_ratio)

        if split == 'train':
            self.token_ids = all_token_ids[:split_idx + seq_len] # +seq_len para garantir que a última sequência de treino seja completa
            self.num_sequences = split_idx
        elif split == 'val':
            self.token_ids = all_token_ids[split_idx:]
            self.num_sequences = total_possible_sequences - split_idx
        else:
            raise ValueError("O parâmetro 'split' deve ser 'train' ou 'val'.")

        if self.num_sequences <= 0:
            raise ValueError(f"O dataset {split} está vazio após a divisão. "
                             f"Verifique o tamanho do texto e o split_ratio.")

    def __len__(self):
        """
        Retorna o número total de sequências disponíveis no dataset.
        """
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Retorna um par (input_sequence, target_sequence) para um dado índice.

        Args:
            idx (int): O índice da sequência a ser retornada.

        Returns:
            tuple: Um tensor para a sequência de entrada e um tensor para a sequência alvo.
        """
        # A sequência de entrada começa em 'idx' e vai até 'idx + seq_len'
        input_seq = self.token_ids[idx : idx + self.seq_len]
        # A sequência alvo é a mesma, mas deslocada em um token para a direita
        # (o modelo prevê o próximo token)
        target_seq = self.token_ids[idx + 1 : idx + self.seq_len + 1]

        # Converte as listas de IDs para tensores PyTorch
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

def collate_fn(batch):
    """
    Função de colagem para o DataLoader. Lida com o padding se necessário,
    mas como estamos usando sequências de comprimento fixo, ela apenas
    empilha os tensores.
    """
    input_sequences, target_sequences = zip(*batch)
    return torch.stack(input_sequences), torch.stack(target_sequences)

# Exemplo de uso (para teste)
if __name__ == "__main__":
    TEXT_FILE = "C:\\Users\\Casa\\Documents\\Nexus\\data.txt" # Caminho absoluto
    TOKENIZER_DIR = "C:\\Users\\Casa\\Documents\\Nexus\\nexus_tokenizer" # Caminho absoluto
    SEQ_LEN = 128

    try:
        dataset = NexusDataset(TEXT_FILE, TOKENIZER_DIR, SEQ_LEN)
        print(f"Dataset criado com {len(dataset)} sequências.")
        print(f"Tamanho do vocabulário: {dataset.vocab_size}")

        # Cria um DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

        # Testa um batch
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            print(f"\nBatch {batch_idx+1}:")
            print(f"Inputs shape: {inputs.shape}")
            print(f"Targets shape: {targets.shape}")
            print(f"Inputs (primeira sequência): {inputs[0].tolist()}")
            print(f"Targets (primeira sequência): {targets[0].tolist()}")
            # Decodifica para ver o texto original (apenas para depuração)
            decoded_input = dataset.tokenizer.decode(inputs[0].tolist())
            decoded_target = dataset.tokenizer.decode(targets[0].tolist())
            print(f"Decoded Input: {decoded_input}")
            print(f"Decoded Target: {decoded_target}")
            if batch_idx == 0: # Apenas um batch para teste
                break

    except ValueError as e:
        print(f"Erro ao criar o Dataset: {e}")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
