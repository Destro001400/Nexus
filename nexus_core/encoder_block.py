
import torch.nn as nn
from nexus_core.multi_head_attention import MultiHeadAttention
from nexus_core.feed_forward import PositionwiseFeedForward
from nexus_core.normalization import RMSNorm

class EncoderBlock(nn.Module):
    """
    Implementa um Bloco Encoder completo do Transformer.

    Um bloco encoder consiste em duas sub-camadas principais:
    1. Uma camada de Multi-Head Self-Attention.
    2. Uma Rede Feed-Forward (Position-wise Feed-Forward).

    Cada uma dessas sub-camadas tem uma conexão residual (Add) em torno dela,
    seguida por uma normalização de camada (Norm).
    """
    def __init__(self, d_model, n_head, d_k, d_v, d_hid, dropout=0.1):
        """
        Inicializador do bloco.

        Args:
            d_model (int): A dimensionalidade do modelo (e das entradas/saídas).
            n_head (int): O número de cabeças de atenção.
            d_k (int): A dimensionalidade das keys e queries.
            d_v (int): A dimensionalidade dos values.
            d_hid (int): A dimensionalidade da camada oculta na rede feed-forward.
            dropout (float, optional): Taxa de dropout. Defaults to 0.1.
        """
        super().__init__()

        # A primeira sub-camada: Multi-Head Attention
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        # A segunda sub-camada: Rede Feed-Forward
        self.pos_ffn = PositionwiseFeedForward(d_model, d_hid, dropout=dropout)

        # Camadas de normalização (usando RMSNorm), uma para cada sub-camada
        self.layer_norm1 = RMSNorm(d_model, eps=1e-6)
        self.layer_norm2 = RMSNorm(d_model, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        """
        Executa a passagem para a frente (forward pass).

        Args:
            enc_input (torch.Tensor): Entrada para o bloco encoder.
                                      Shape: [batch_size, seq_len, d_model]
            slf_attn_mask (torch.Tensor, optional): Máscara para a self-attention.
                                                    Defaults to None.

        Returns:
            enc_output (torch.Tensor): Saída do bloco encoder.
                                       Shape: [batch_size, seq_len, d_model]
            slf_attn (torch.Tensor): Pesos de atenção da camada de self-attention.
        """

        # --- Início da primeira sub-camada (Multi-Head Attention) ---

        # Salva a entrada original para a conexão residual (o "Add")
        residual = enc_input

        # Passa a entrada pela camada de Multi-Head Attention.
        # No encoder, Q, K e V são todos iguais à entrada (self-attention).
        enc_output, slf_attn = self.slf_attn(
            q=enc_input, k=enc_input, v=enc_input, mask=slf_attn_mask)

        # Aplica dropout à saída da atenção
        enc_output = self.dropout(enc_output)

        # Aplica a conexão residual ("Add") e a normalização ("Norm")
        enc_output = self.layer_norm1(residual + enc_output)

        # --- Fim da primeira sub-camada ---


        # --- Início da segunda sub-camada (Feed-Forward) ---

        # Salva a saída da primeira sub-camada para a próxima conexão residual
        residual = enc_output

        # Passa o resultado pela rede Feed-Forward
        enc_output = self.pos_ffn(enc_output)

        # Aplica dropout
        enc_output = self.dropout(enc_output)

        # Aplica a segunda conexão residual ("Add") e normalização ("Norm")
        enc_output = self.layer_norm2(residual + enc_output)

        # --- Fim da segunda sub-camada ---

        return enc_output, slf_attn
