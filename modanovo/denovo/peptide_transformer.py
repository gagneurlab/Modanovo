"""Transformer models for peptides."""
import torch
import einops

from . import FloatEncoder, PositionalEncoder
      

class DanielasPeptideDecoder(torch.nn.Module):
    """A transformer decoder for peptide sequences.

    Parameters
    ----------
    n_tokens : int or Tokenizer
        The number of tokens used to tokenize molecular sequences.
    d_model : int, optional
        The latent dimensionality to represent elements of the sequence.
    nhead : int, optional
        The number of attention heads in each layer. ``d_model`` must be
        divisible by ``nhead``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    positional_encoder : PositionalEncoder or bool, optional
        The positional encodings to use for the molecular sequence. If
        ``True``, the default positional encoder is used. ``False`` disables
        positional encodings, typically only for ablation tests.
    padding_int : int, optional
        The index that represents padding in the input sequence. Required
        only if ``n_tokens`` was provided as an ``int``.
    max_charge : int, optional
        The maximum charge state for peptide sequences.


    """

    def __init__(
        self,
        n_tokens: int,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        pos_encoder: PositionalEncoder | bool = True,
        max_charge: int = 5,
        n_expanded_tokens=None
    ) -> None:
        """Initialize a PeptideDecoder."""
        super().__init__()

        if callable(pos_encoder):
            self.pos_encoder = pos_encoder
        elif pos_encoder:
            self.pos_encoder = PositionalEncoder(d_model)
        else:
            self.pos_encoder = torch.nn.Identity()

        self.charge_encoder = torch.nn.Embedding(max_charge, d_model)
        
        # Additional model components
        self.mass_encoder = FloatEncoder(d_model)
        
        layer = torch.nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_decoder = torch.nn.TransformerDecoder(
            layer,
            num_layers=n_layers,
        )

        
        self.aa_encoder = torch.nn.Embedding( n_tokens + 1,  d_model, padding_idx=0)
        self.final = torch.nn.Linear( d_model, self.aa_encoder.num_embeddings)
        
        
        if n_expanded_tokens is not None:
            self.aa_encoder_expanded = torch.nn.Embedding(n_tokens + 1 + n_expanded_tokens,
                                                           d_model, padding_idx=0)
            
            self.final_expanded = torch.nn.Linear(d_model, self.aa_encoder_expanded.num_embeddings)
        
        self.n_tokens = n_tokens
        self.n_expanded_tokens = n_expanded_tokens if n_expanded_tokens is not None else 0
        self.d_model = d_model
        print(f"[DECODER] n_tokens {self.n_tokens}, n_expanded_tokens {self.n_expanded_tokens}")
        
    def _expand_tokens(self):
        if self.n_expanded_tokens==0:
            return
        total_tokens = self.n_tokens + self.n_expanded_tokens
        # Add vocab
        self.aa_encoder_expanded.weight = torch.nn.Parameter(
                        torch.cat((self.aa_encoder.weight, 
                                        #torch.randn(self.n_expanded_tokens, self.d_model, device=self.device) 
                            einops.repeat(self.aa_encoder.weight.mean(dim=0), 'm ->  k m ', k=self.n_expanded_tokens)
                                              ) ) )
        
        self.final_expanded.weight = torch.nn.Parameter(
                                torch.cat((self.final.weight, 
                                    #torch.randn(self.n_expanded_tokens, self.d_model, device=self.device) \
                            einops.repeat(self.final.weight.mean(dim=0), 'm ->  k m ', k=self.n_expanded_tokens)  
                                           
                                          ) ) )
        
        self.final_expanded.bias = torch.nn.Parameter(
                                        torch.cat((self.final.bias, 
                                                   #torch.randn(self.n_expanded_tokens, device=self.device) 
                                                   self.final.bias.mean().repeat(self.n_expanded_tokens)
                                                  ) ) )
        
        ## Replace old with new
        self.aa_encoder = self.aa_encoder_expanded
        self.final = self.final_expanded
        self.final_expanded = None
        self.aa_encoder_expanded = None
        self.n_tokens = total_tokens
        self.n_expanded_tokens = 0         
        
        
    def forward(
        self,
        tokens: torch.Tensor | None,
        precursors: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the next amino acid for a collection of sequences.

        Parameters
        ----------
        tokens : list of str, torch.Tensor, or None
            The partial peptide sequences for which to predict the next
            amino acid. Optionally, these may be the token indices instead
            of a string.
        precursors : torch.Tensor of size (batch_size, 2)
            The precursor mass (axis 0) and charge (axis 1).
        memory : torch.Tensor of shape (batch_size, n_peaks, d_model)
            The representations from a ``TransformerEncoder``, such as a
            ``SpectrumEncoder``.
        memory_key_padding_mask : torch.Tensor of shape (batch_size, n_peaks)
            The mask that indicates which elements of ``memory`` are padding.

        Returns
        -------
        scores : torch.Tensor of size (batch_size, len_sequence, n_amino_acids)
            The raw output for the final linear layer. These can be Softmax
            transformed to yield the probability of each amino acid for the
            prediction.

        """
        # Prepare sequences
        #if tokens is None:
        #    #tokens = torch.tensor([[]], dtype=torch.long, device=self.device)
        #    batch_size = precursors.shape[0]
         #   tokens = torch.zeros((batch_size, 0 ), device=self.device, dtype=torch.long )
            
        # Encode everything:
        masses = self.mass_encoder(precursors[:, None, 0]) # self.mass_encoder(precursors[:, None, [0]])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]

        # Feed through model:
        if tokens is not None:
            tokens = self.aa_encoder(tokens)
            tgt = torch.cat([precursors, tokens], dim=1)
        else:
            tgt = precursors
            
        tgt_key_padding_mask = tgt.sum(axis=2) == 0
        tgt = self.pos_encoder(tgt)
        tgt_mask = generate_tgt_mask(tgt.shape[1]).to(self.device)
        preds = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.final(preds)
    
    @property
    def device(self) -> torch.device:
        """The current device for the model."""
        return next(self.parameters()).device


def generate_tgt_mask(sz: int) -> torch.Tensor:
    """Generate a square mask for the sequence.

    Parameters
    ----------
    sz : int
        The length of the target sequence.
    """
    return ~torch.triu(torch.ones(sz, sz, dtype=torch.bool)).transpose(0, 1)
