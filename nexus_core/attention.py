
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    """
    Implementa a atenção de produto escalar em escala (Scaled Dot-Product Attention).

    Esta é a engrenagem central do mecanismo de atenção do Transformer. Ela calcula
    pesos de atenção para uma sequência de entrada, permitindo que o modelo foque
    em partes mais relevantes da sequência ao processar uma determinada posição.
    """
    def __init__(self, temperature, attn_dropout=0.1):
        """
        Inicializador da camada.

        Args:
            temperature (float): Fator de escala para a energia de atenção.
                               É a raiz quadrada da dimensão das chaves (sqrt(d_k)).
            attn_dropout (float, optional): Taxa de dropout a ser aplicada nos
                                            pesos de atenção. Defaults to 0.1.
        """
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """
        Executa a passagem para a frente (forward pass).

        Args:
            q (torch.Tensor): Tensor de Queries (consultas).
                              Shape: [batch_size, n_head, len_q, d_k]
            k (torch.Tensor): Tensor de Keys (chaves).
                              Shape: [batch_size, n_head, len_k, d_k]
            v (torch.Tensor): Tensor de Values (valores).
                              Shape: [batch_size, n_head, len_v, d_v]
                              (len_k deve ser igual a len_v)
            mask (torch.Tensor, optional): Máscara para impedir a atenção a
                                           certas posições (ex: padding ou
                                           posições futuras em decodificadores).
                                           Shape: [batch_size, 1, 1, len_k]
                                           Defaults to None.

        Returns:
            output (torch.Tensor): O tensor de saída após a aplicação da atenção.
                                   Shape: [batch_size, n_head, len_q, d_v]
            attn (torch.Tensor): Os pesos de atenção calculados.
                                 Shape: [batch_size, n_head, len_q, len_k]
        """
        # 1. Calcula a pontuação de atenção (energia)
        # Matriz de multiplicação entre Queries (q) e Chaves transpostas (k.transpose)
        # O resultado (attn) mede a compatibilidade entre cada query e cada chave.
        # Shape: [batch_size, n_head, len_q, len_k]
        attn = torch.matmul(q, k.transpose(2, 3))

        # 2. Escala a pontuação
        # Divide pela temperatura (sqrt(d_k)) para estabilizar o gradiente.
        attn = attn / self.temperature

        # 3. Aplica a máscara (se existir)
        # Onde a máscara for True, substitui a pontuação por um valor muito pequeno (-1e9).
        # Isso faz com que o softmax atribua probabilidade próxima de zero a essas posições.
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        # 4. Aplica o Softmax
        # Converte as pontuações de atenção em probabilidades (pesos que somam 1).
        # Shape: [batch_size, n_head, len_q, len_k]
        attn = self.softmax(attn)

        # 5. Aplica o Dropout
        # Zera aleatoriamente alguns pesos de atenção para regularização.
        attn = self.dropout(attn)

        # 6. Calcula a saída
        # Multiplica os pesos de atenção (attn) pelos Valores (v).
        # Isso pondera os valores, dando mais importância àqueles com maior atenção.
        # Shape: [batch_size, n_head, len_q, d_v]
        output = torch.matmul(attn, v)

        return output, attn
