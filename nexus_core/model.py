
import torch
import torch.nn as nn
from nexus_core.positional_encoding import PositionalEncoding
from nexus_core.decoder_block import DecoderBlock

class NexusModel(nn.Module):
    """
    A implementação completa do modelo Nexus, com arquitetura GPT-style (Decoder-only).
    """
    def __init__(self, n_token, d_model, n_layer, n_head, d_k, d_v, d_hid, dropout, max_len=5000):
        """
        Inicializador do modelo completo.

        Args:
            n_token (int): Tamanho do vocabulário (número de tokens).
            d_model (int): Dimensionalidade do modelo.
            n_layer (int): Número de DecoderBlocks a serem empilhados.
            n_head (int): Número de cabeças de atenção.
            d_k (int): Dimensionalidade das keys e queries.
            d_v (int): Dimensionalidade dos values.
            d_hid (int): Dimensionalidade da camada oculta na rede feed-forward.
            dropout (float): Taxa de dropout.
            max_len (int, optional): Comprimento máximo da sequência. Defaults to 5000.
        """
        super().__init__()

        # Camada de Embedding: converte tokens de entrada (índices) em vetores densos.
        self.token_embedding = nn.Embedding(n_token, d_model)

        # Camada de Codificação Posicional: injeta a informação de ordem.
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)

        # Pilha de Blocos Decodificadores (Decoder Stack)
        # Cria `n_layer` cópias do nosso DecoderBlock.
        self.layer_stack = nn.ModuleList([
            DecoderBlock(d_model, n_head, d_k, d_v, d_hid, dropout=dropout)
            for _ in range(n_layer)])

        # Camada linear final: projeta a saída do decoder de volta para o espaço do vocabulário.
        # O resultado são os logits para cada token no vocabulário.
        self.output_projection = nn.Linear(d_model, n_token)

        self.dropout = nn.Dropout(p=dropout)

        # Inicialização de pesos (uma boa prática)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt_seq, tgt_mask):
        """
        Executa a passagem para a frente (forward pass) do modelo completo.

        Args:
            tgt_seq (torch.Tensor): A sequência de tokens de entrada (alvo).
                                    Shape: [batch_size, seq_len]
            tgt_mask (torch.Tensor): A máscara de atenção para a sequência alvo.
                                     Shape: [batch_size, seq_len, seq_len]

        Returns:
            torch.Tensor: Os logits de saída para cada posição na sequência.
                          Shape: [batch_size, seq_len, n_token]
        """
        # 1. Embedding e Codificação Posicional
        # Converte a sequência de entrada em embeddings e adiciona a posição.
        dec_output = self.token_embedding(tgt_seq)
        dec_output = self.positional_encoding(dec_output)
        dec_output = self.dropout(dec_output)

        # 2. Passa pela Pilha de Decodificadores
        # Itera por cada DecoderBlock na nossa pilha.
        for dec_layer in self.layer_stack:
            dec_output, slf_attn = dec_layer(dec_output, tgt_attn_mask=tgt_mask)

        # 3. Projeção Final
        # Passa a saída final do decoder pela camada linear para obter os logits.
        logits = self.output_projection(dec_output)

        return logits
