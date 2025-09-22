
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implementa a Codificação Posicional (Positional Encoding).

    Como o modelo Transformer em si não tem noção da ordem das palavras,
    injetamos essa informação nos embeddings de entrada. A codificação posicional
    adiciona um vetor ao embedding de cada palavra, onde o vetor depende da
    posição da palavra na sequência.

    Usa a fórmula com seno e cosseno do paper original "Attention Is All You Need".
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Inicializador da camada.

        Args:
            d_model (int): A dimensionalidade do modelo (deve ser a mesma dos embeddings).
            dropout (float, optional): Taxa de dropout. Defaults to 0.1.
            max_len (int, optional): O comprimento máximo da sequência suportada.
                                     Defaults to 5000.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Cria uma matriz `pe` (positional encoding) de shape [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # Cria um tensor de posições: [0, 1, 2, ..., max_len-1]
        # Shape: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calcula o termo de divisão para as frequências de seno e cosseno.
        # div_term é calculado em log-space para estabilidade numérica.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Aplica a fórmula de seno para as colunas pares (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Aplica a fórmula de cosseno para as colunas ímpares (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Adiciona uma dimensão de batch no início do tensor `pe`.
        # Shape final: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Registra `pe` como um buffer. Buffers são parte do estado do modelo,
        # mas não são considerados parâmetros a serem treinados.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Executa a passagem para a frente (forward pass).

        Args:
            x (torch.Tensor): Os embeddings de entrada para a sequência.
                              Shape: [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: Os embeddings com a informação posicional adicionada.
                          Shape: [batch_size, seq_len, d_model]
        """
        # Adiciona a codificação posicional aos embeddings de entrada.
        # O fatiamento `self.pe[:, :x.size(1)]` garante que a codificação
        # tenha o mesmo comprimento da sequência de entrada `x`.
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
