
import torch.nn as nn
from nexus_core.multi_head_attention import MultiHeadAttention
from nexus_core.feed_forward import PositionwiseFeedForward
from nexus_core.normalization import RMSNorm

class DecoderBlock(nn.Module):
    """
    Implementa um Bloco Decoder completo do Transformer (estilo GPT).

    Um bloco decoder para um modelo auto-regressivo (como o GPT) é muito
    semelhante a um bloco encoder. Ele consiste em:
    1. Uma camada de Multi-Head Self-Attention (mascarada).
    2. Uma Rede Feed-Forward (Position-wise Feed-Forward).

    A diferença crucial é que a camada de self-attention deve ser mascarada
    para impedir que uma posição "veja" as posições futuras.
    """
    def __init__(self, d_model, n_head, d_k, d_v, d_hid, dropout=0.1):
        """
        Inicializador do bloco.

        Args:
            d_model (int): A dimensionalidade do modelo.
            n_head (int): O número de cabeças de atenção.
            d_k (int): A dimensionalidade das keys e queries.
            d_v (int): A dimensionalidade dos values.
            d_hid (int): A dimensionalidade da camada oculta na rede feed-forward.
            dropout (float, optional): Taxa de dropout. Defaults to 0.1.
        """
        super().__init__()
        # Primeira sub-camada: Multi-Head Self-Attention (que será mascarada)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        # Segunda sub-camada: Rede Feed-Forward
        self.pos_ffn = PositionwiseFeedForward(d_model, d_hid, dropout=dropout)

        # Camadas de normalização (usando RMSNorm)
        self.layer_norm1 = RMSNorm(d_model, eps=1e-6)
        self.layer_norm2 = RMSNorm(d_model, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, tgt_attn_mask=None):
        """
        Executa a passagem para a frente (forward pass).

        Args:
            dec_input (torch.Tensor): Entrada para o bloco decoder.
                                      Shape: [batch_size, seq_len, d_model]
            tgt_attn_mask (torch.Tensor, optional): Máscara para a self-attention,
                                                    para impedir a visão do futuro.
                                                    Defaults to None.

        Returns:
            dec_output (torch.Tensor): Saída do bloco decoder.
                                       Shape: [batch_size, seq_len, d_model]
            slf_attn (torch.Tensor): Pesos de atenção da camada de self-attention.
        """

        # --- Primeira sub-camada (Masked Multi-Head Attention) ---
        residual = dec_input
        dec_output, slf_attn = self.slf_attn(
            q=dec_input, k=dec_input, v=dec_input, mask=tgt_attn_mask)
        dec_output = self.dropout(dec_output)
        dec_output = self.layer_norm1(residual + dec_output)

        # --- Segunda sub-camada (Feed-Forward) ---
        residual = dec_output
        dec_output = self.pos_ffn(dec_output)
        dec_output = self.dropout(dec_output)
        dec_output = self.layer_norm2(residual + dec_output)

        return dec_output, slf_attn
