import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CodeTestRelationModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        heads,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.device = device
        self.embedding = TextEmbedding(vocab_size, embedding_dim, device)
        self.file_encoder = BiGRUEncoder(embedding_dim, hidden_dim)
        self.test_encoder = BiGRUEncoder(embedding_dim, hidden_dim)
        self.multi_head_attention = MultiHeadAttention(
            hidden_dim * 2, hidden_dim * 2, heads
        )
        self.mlp = MLP(hidden_dim * 2, hidden_dim, output_dim)

    def forward(self, file_tokens, file_lengths, test_tokens, test_lengths):
        # File Encoding
        file_embeddings = self.embedding(file_tokens)
        file_encodings, _ = self.file_encoder(file_embeddings, file_lengths)

        test_embeddings = self.embedding(test_tokens)
        test_encodings, _ = self.test_encoder(test_embeddings, test_lengths)

        attention_output, _ = self.multi_head_attention(
            file_encodings, test_encodings, test_encodings
        )

        attention_output_avg = attention_output.mean(dim=1)
        output = self.mlp(attention_output_avg)

        return output


class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, device):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, device=device)

    def forward(self, x):
        return self.embedding(x)


class BiGRUEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(
            embedding_dim, hidden_dim, bidirectional=True, batch_first=True
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.gru(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output, hidden


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim_queries, embed_dim_keys, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim_queries = embed_dim_queries
        self.embed_dim_keys = embed_dim_keys

        assert (
            embed_dim_queries % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads."
        assert (
            embed_dim_keys % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads."

        self.head_dim_queries = embed_dim_queries // num_heads
        self.head_dim_keys = embed_dim_keys // num_heads

        self.linear_queries = nn.Linear(embed_dim_queries, embed_dim_queries)
        self.linear_keys = nn.Linear(embed_dim_keys, embed_dim_keys)
        self.linear_values = nn.Linear(embed_dim_keys, embed_dim_keys)
        self.output_linear = nn.Linear(embed_dim_queries, embed_dim_queries)

    def split_heads(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.head_dim_queries)
        return x.transpose(1, 2)  # (B, num_heads, L, D)

    def forward(self, queries, keys, values):
        batch_size = queries.shape[0]

        queries = self.linear_queries(queries)  # (B, L_q, D)
        keys = self.linear_keys(keys)  # Shape: (B, L_k, dim)
        values = self.linear_values(values)  # (B, L_k, D)

        queries = self.split_heads(queries)
        keys = self.split_heads(keys)
        values = self.split_heads(values)

        keys_transposed = keys.transpose(-2, -1)
        attention_scores = torch.matmul(queries, keys_transposed) / (
            self.head_dim_keys**0.5
        )  # (B, num_heads, L_q, L_k)

        attention_weights = F.softmax(
            attention_scores, dim=-1
        )  # (B, num_heads,L_q, L_k)

        output = torch.matmul(
            attention_weights, values
        )  # (B, num_heads, L_q, head_dim_keys)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.embed_dim_queries)
        )  # (B, L_q, D)

        output = self.output_linear(output)  # (B, L_q, D)

        return output, attention_weights


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dp1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dp2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dp1(x)
        x = F.relu(self.fc1(x))
        x = self.dp2(x)
        x = self.fc2(x)
        return x
