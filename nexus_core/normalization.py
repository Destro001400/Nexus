
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Implementa a Normalização Root Mean Square (RMSNorm).

    É uma alternativa mais simples e computacionalmente eficiente ao LayerNorm.
    A principal diferença é que o RMSNorm apenas normaliza a saída com base na
    escala (a raiz quadrada da média dos quadrados), sem recentralizar os dados.
    """
    def __init__(self, d_model, eps=1e-6):
        """
        Inicializador da camada.

        Args:
            d_model (int): A dimensionalidade do modelo.
            eps (float, optional): Um pequeno valor adicionado ao denominador para
                                   evitar divisão por zero. Defaults to 1e-6.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # O `gain` (ganho) é um parâmetro de escala que o modelo aprende.
        self.gain = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        Executa a passagem para a frente (forward pass).

        Args:
            x (torch.Tensor): O tensor de entrada.

        Returns:
            torch.Tensor: O tensor normalizado.
        """
        # Calcula a raiz quadrada da média dos quadrados ao longo da última dimensão.
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normaliza a entrada e aplica o ganho.
        return (x / rms) * self.gain
