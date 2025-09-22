
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    Implementa a rede Feed-Forward (Position-wise Feed-Forward Network).

    Dentro de um bloco Transformer, esta rede é aplicada a cada posição (token)
    separadamente e de forma idêntica. Consiste em duas transformações lineares
    com uma função de ativação não-linear entre elas.
    """
    def __init__(self, d_in, d_hid, dropout=0.1):
        """
        Inicializador da camada.

        Args:
            d_in (int): Dimensionalidade de entrada (e saída). Geralmente é a d_model.
            d_hid (int): Dimensionalidade da camada oculta (interna).
            dropout (float, optional): Taxa de dropout. Defaults to 0.1.
        """
        super().__init__()
        # A arquitetura padrão usa uma camada linear para expandir a dimensão,
        # uma função de ativação (ReLU), e outra camada linear para contrair de volta.
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Executa a passagem para a frente (forward pass).

        Args:
            x (torch.Tensor): O tensor de entrada.
                              Shape: [batch_size, seq_len, d_in]

        Returns:
            torch.Tensor: O tensor de saída.
                          Shape: [batch_size, seq_len, d_in]
        """
        # Passa pela primeira camada linear e aplica a função de ativação ReLU
        output = self.relu(self.w_1(x))
        # Aplica dropout para regularização
        output = self.dropout(output)
        # Passa pela segunda camada linear
        output = self.w_2(output)

        return output
