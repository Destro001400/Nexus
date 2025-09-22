
import torch
import torch.nn as nn
from nexus_core.attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    """
    Implementa a camada de Atenção de Múltiplas Cabeças (Multi-Head Attention).

    Esta camada permite que o modelo preste atenção conjuntamente a informações
    de diferentes subespaços de representação em diferentes posições. Em vez de
    realizar uma única função de atenção, ela projeta as queries, keys e values
    para `n_head` diferentes subespaços e aplica a atenção em paralelo em cada um.
    Os resultados são então concatenados e projetados novamente para a saída final.
    """
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        """
        Inicializador da camada.

        Args:
            n_head (int): O número de cabeças de atenção paralelas.
            d_model (int): A dimensionalidade do modelo (espaço de embedding).
            d_k (int): A dimensionalidade das queries e keys.
            d_v (int): A dimensionalidade dos values.
            dropout (float, optional): Taxa de dropout. Defaults to 0.1.
        """
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # Camadas lineares para projetar as entradas Q, K, V para cada cabeça
        # Note: A saída de todas as cabeças é concatenada, então a dimensão total
        # é n_head * d_k (ou d_v).
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        # Camada de atenção que criamos anteriormente
        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        # Camada linear final para projetar a saída concatenada das cabeças
        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Executa a passagem para a frente (forward pass).

        Args:
            q (torch.Tensor): Tensor de Queries. Shape: [batch_size, len_q, d_model]
            k (torch.Tensor): Tensor de Keys. Shape: [batch_size, len_k, d_model]
            v (torch.Tensor): Tensor de Values. Shape: [batch_size, len_v, d_model]
            mask (torch.Tensor, optional): Máscara de atenção. Defaults to None.

        Returns:
            output (torch.Tensor): O tensor de saída. Shape: [batch_size, len_q, d_model]
            attn (torch.Tensor): Os pesos de atenção. Shape: [batch_size, n_head, len_q, len_k]
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, len_q, _ = q.size()
        batch_size, len_k, _ = k.size()
        batch_size, len_v, _ = v.size()

        # 1. Projeção Linear: Passa as entradas Q, K, V pelas camadas lineares.
        # Shape: [batch_size, len, n_head * d_k_or_d_v]
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        # 2. Divide em Múltiplas Cabeças: Reorganiza os tensores para o formato de múltiplas cabeças.
        # O view() e transpose() separam a dimensão (n_head * d_k) em (n_head, d_k).
        # Shape final: [batch_size, n_head, len, d_k_or_d_v]
        q = q.view(batch_size, len_q, n_head, d_k).transpose(1, 2)
        k = k.view(batch_size, len_k, n_head, d_k).transpose(1, 2)
        v = v.view(batch_size, len_v, n_head, d_v).transpose(1, 2)

        # 3. Aplica a Atenção: Passa as cabeças para a nossa camada ScaledDotProductAttention.
        # A máscara é transmitida (broadcasted) para todas as cabeças.
        # output shape: [batch_size, n_head, len_q, d_v]
        # attn shape:   [batch_size, n_head, len_q, len_k]
        output, attn = self.attention(q, k, v, mask=mask)

        # 4. Concatena as Cabeças: Reverte a operação de divisão.
        # Transpõe para juntar as cabeças e depois usa contiguous() e view() para
        # achatar a dimensão das cabeças e dos valores.
        # Shape: [batch_size, len_q, n_head * d_v]
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, -1)

        # 5. Projeção Final: Passa o resultado pela camada linear final.
        # Shape: [batch_size, len_q, d_model]
        output = self.dropout(self.fc(output))

        return output, attn
