"""A de novo peptide sequencing model."""

import collections
import heapq
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import einops
import torch
import numpy as np
import lightning.pytorch as pl
import pandas as pd

from . import evaluate
from . import PeakEncoder, DanielasSpectrumEncoder, DanielasPeptideDecoder
from ..data import ms_io
from datetime import datetime

logger = logging.getLogger("modanovo")

class Spec2Pep(pl.LightningModule):
    """
    A Transformer model for de novo peptide sequencing.

    Use this model in conjunction with a pytorch-lightning Trainer.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality used by the transformer model.
    n_head : int
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int
        The dimensionality of the fully connected layers in the transformer
        model.
    n_layers : int
        The number of transformer layers.
    dropout : float
        The dropout probability for all layers.
    dim_intensity : Optional[int]
        The number of features to use for encoding peak intensity. The remaining
        (``dim_model - dim_intensity``) are reserved for encoding the m/z value.
        If ``None``, the intensity will be projected up to ``dim_model`` using a
        linear layer, then summed with the m/z encoding for each peak.
    max_length : int
        The maximum peptide length to decode.
    max_charge : int
        The maximum precursor charge to consider.
    precursor_mass_tol : float, optional
        The maximum allowable precursor mass tolerance (in ppm) for correct
        predictions.
    isotope_error_range : Tuple[int, int]
        Take into account the error introduced by choosing a non-monoisotopic
        peak for fragmentation by not penalizing predicted precursor m/z's that
        fit the specified isotope error:
        `abs(calc_mz - (precursor_mz - isotope * 1.00335 / precursor_charge))
        < precursor_mass_tol`
    min_peptide_len : int
        The minimum length of predicted peptides.
    n_beams: int
        Number of beams used during beam search decoding.
    top_match: int
        Number of PSMs to return for each spectrum.
    n_log : int
        The number of epochs to wait between logging messages.
    train_label_smoothing: float
        Smoothing factor when calculating the training loss.
    warmup_iters: int
        The number of warm up iterations for the learning rate scheduler.
    max_iters: int
        The total number of iterations for the learning rate scheduler.
    out_writer: Optional[str]
        The output writer for the prediction results.
    calculate_precision: bool
        Calculate the validation set precision during training.
        This is expensive.
    tb_summarywriter
    **kwargs : Dict
        Additional keyword arguments passed to the Adam optimizer.
    """

    def __init__(
        self,
        dim_model: int = 512,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 9,
        dropout: float = 0.0,
        dim_intensity: Optional[int] = None,
        max_length: int = 100,
        residues: Optional[Dict[str, float]] = None,
        max_charge: int = 5,
        precursor_mass_tol: float = 50,
        isotope_error_range: Tuple[int, int] = (0, 1),
        min_peptide_len: int = 6,
        n_beams: int = 1,
        top_match: int = 1,
        n_log: int = 10,
        train_label_smoothing: float = 0.01,
        warmup_iters: int = 100_000,
        max_iters: int = 600_000,
        tokenizer = None,
        out_writer: Optional[ms_io.MztabWriter] = None,
        calculate_precision: bool = False,
        train_from_default_residues:bool = False,
        tb_summarywriter = None,
        lr: float = 0.05,
        weight_decay: float= 1e-05,

        **kwargs: Dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer) # tokenizer already contains stop token 
        if train_from_default_residues:
            n_tokens = self.tokenizer.n_default_residues
            n_expanded_tokens = self.tokenizer.n_expanded_residues
        else:
            n_tokens = self.vocab_size
            n_expanded_tokens = None
            
        
        # Build the model.
        self.encoder = DanielasSpectrumEncoder(
            d_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            dim_intensity=dim_intensity,
        )
        self.decoder = DanielasPeptideDecoder(
            n_tokens=n_tokens,
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            max_charge=max_charge,
            n_expanded_tokens=n_expanded_tokens
        )
        
        self.softmax = torch.nn.Softmax(2)
        self.celoss = torch.nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=train_label_smoothing
        )
        self.val_celoss = torch.nn.CrossEntropyLoss(ignore_index=0)
        # Optimizer settings.
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.lr= lr
        self.weight_decay = weight_decay
        self.opt_kwargs = kwargs

        # Data properties.
        self.max_length = max_length
        self.precursor_mass_tol = precursor_mass_tol
        self.isotope_error_range = isotope_error_range
        self.min_peptide_len = min_peptide_len
        self.n_beams = n_beams
        self.top_match = top_match
        self.stop_token = self.tokenizer.stop_int

        # Logging.
        self.calculate_precision = calculate_precision
        self.n_log = n_log
        self._history = []
        
        self.out_writer = out_writer
        self._history = []

    @property
    def history(self):
        """The training history of a model."""
        return pd.DataFrame(self._history)

    @property
    def n_parameters(self):
        """The number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def setup(self, stage):
        
        if self.decoder.n_expanded_tokens>0:
            logger.info('Expanding embedding and final layers')
            logger.info(f'Weigths before expansion {self.decoder.aa_encoder.weight.shape}')

            self.decoder._expand_tokens()

            logger.info(f'Weigths after expansion {self.decoder.aa_encoder.weight.shape}')
            
            
    def _process_batch(self, batch):
        """ Prepare batch returned from AnnotatedSpectrumDataset of the 
            latest depthcharge version
        
        Each batch is a dict and contains these keys: 
             ['peak_file', 'scan_id', 'ms_level', 'precursor_mz',
             'precursor_charge', 'mz_array', 'intensity_array',
             'seq']
        Returns
        -------
        spectra : torch.Tensor of shape (batch_size, n_peaks, 2)
            The padded mass spectra tensor with the m/z and intensity peak values
            for each spectrum.
        precursors : torch.Tensor of shape (batch_size, 3)
            A tensor with the precursor neutral mass, precursor charge, and
            precursor m/z.
        seqs : np.ndarray
            The spectrum identifiers (during de novo sequencing) or peptide
            sequences (during training).

        """
        precursor_mzs = batch["precursor_mz"]
        precursor_charges = batch["precursor_charge"]
        precursor_masses = (precursor_mzs - 1.007276) * precursor_charges
        precursors = torch.vstack([precursor_masses, precursor_charges, 
                                 precursor_mzs
                                ] ).T.float()
        
        mzs, ints = batch['mz_array'], batch['intensity_array']
        #spectra = torch.dstack([mzs, ints] ).float()
        spectra = torch.stack([mzs, ints], dim=2)
        
        seqs = batch['seq'] if "seq" in batch else None# already tokenized
        
        return spectra, precursors, seqs
        #return batch['mz_array'], batch['intensity_array'], precursors, seqs
    
    def forward( self, batch ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Predict peptide sequences for a batch of MS/MS spectra.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        pred_peptides : List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide predictions. A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        spectra, precursors, _ = self._process_batch(batch)
        return self.beam_search_decode( spectra, precursors ) 
    
    def beam_search_decode(
        self, spectra: torch.Tensor, precursors: torch.Tensor
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Beam search decoding of the spectrum predictions.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        pred_peptides : List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide prediction(s). A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        memories, mem_masks = self.encoder(spectra)
        # Get tokens to discard from predictions 
        discarded = self.tokenizer.discarded_residue_predictions
        
        # Sizes.
        batch = spectra.shape[0]  # B
        length = self.max_length + 1  # L
        vocab = self.vocab_size + 1  # V
        beam = self.n_beams  # S

        # Initialize scores and tokens.
        scores = torch.full(
            size=(batch, length, vocab, beam), fill_value=torch.nan
        )
        scores = scores.type_as(memories)
        tokens = torch.zeros(batch, length, beam, dtype=torch.long, device=self.decoder.device)

        # Create cache for decoded beams.
        pred_cache = collections.OrderedDict((i, []) for i in range(batch))

        # Get the first prediction.
        pred = self.decoder(None, precursors, memories, mem_masks)
        pred[:, :, discarded] = -10 #torch.min(pred).item()

        '''
        SCALE_WEIGHTS = torch.tensor([  0.72070163,   1.905072  ,   1.7493458 ,   1.7917151 ,
                  1.7960036 ,   1.6450799 ,   1.6421816 ,   1.2478116 ,
                  1.8076913 ,   0.72070163,   1.4937607 ,   1.5879427 ,
                  1.4798287 ,   1.2087604 ,   1.5804187 ,   1.2530442 ,
                  1.4598985 ,   1.49777   ,   1.3629156 ,   1.4226269 ,
                  1.1897603 ,   1.1422005 ,   0.8642662 ,   0.68754303,
                  1.3098437 ,   0.9083212 ,   0.95902354,   4.0342565 ,
                  1.8366857 ,   1.10889   ,   4.2633743 , 303.34232   ,
                 64.96296   ,   2.9005373 ,   1.1209438 ,   1.0570812 ,
                  1.2258393 ,   1.2629288 ,   1.0598551 ,   1.2963288 ,
                  1.2441365 ,   0.6888115 ,   0.6436954 ,   0.5884731 ,
                  0.65618235,   0.66892713,   0.8488138 ,   0.7832537 ,
                  0.6384949 ,   0.6683687 ,   0.8663416 ,   0.8049453 ,
                  0.9247613 ,   0.7530768 ])
        
        SCALE_BIASES = torch.tensor([-60.753857  ,   6.13396   ,   5.921174  ,   6.128576  ,
                  6.0527782 ,   5.704969  ,   5.6413965 ,   4.4473376 ,
                  6.324111  , -60.753857  ,   5.180801  ,   5.6599693 ,
                  5.3306255 ,   4.722912  ,   5.756953  ,   4.273476  ,
                  5.0568004 ,   5.4288573 ,   5.4971848 ,   5.1443057 ,
                  3.9424827 ,   2.765445  ,   2.0601718 ,   1.8537295 ,
                  3.537485  ,   1.5071031 ,   1.0211586 , -58.574425  ,
                  6.7423015 ,   1.5177168 ,   6.8960557 ,   6.761205  ,
                  3.2829773 ,   3.943428  ,   3.8216267 ,   3.248138  ,
                  4.18678   ,   3.6595573 ,   3.2090046 ,   3.3568308 ,
                  4.4246774 ,   0.29574898,   0.28381974,  -0.18679036,
                  0.06388789,   0.18981196,   1.4073281 ,   1.657557  ,
                  0.17945968,   1.713722  ,  -0.9224126 ,   1.8746629 ,
                  2.01269   ,   1.20818   ])

        SCALING_FACTORS = torch.tensor([1.0000e+00, 1.1176e+00, 8.9266e-01, 1.2799e+00, 2.0024e+00, 7.8319e-01,
        9.7264e-01, 3.3117e-01, 4.9276e-01, 1.0000e+00, 9.4937e-01, 1.5669e+00,
        1.0172e+00, 7.3539e-01, 1.3796e+00, 2.2828e-02, 1.1795e+00, 5.0710e-01,
        9.9274e-01, 3.6123e-01, 1.3353e-02, 7.1704e+00, 2.2721e-05, 5.1570e-05,
        3.0313e-01, 3.5667e-05, 7.3600e-05, 2.8605e-02, 7.5546e-01, 1.7908e-05,
        6.1054e-07, 6.1185e-07, 1.8786e-05, 1.1614e-05, 2.2244e-06, 3.5602e-06,
        1.0821e+01, 7.2495e-06, 5.3135e+00, 7.2348e-06, 2.6924e-02, 6.3706e-05,
        9.1777e-05, 7.2480e-05, 5.7518e-05, 7.3600e-05, 7.6758e-05, 8.5815e-05,
        6.9488e-05, 6.8705e-05, 9.5566e-05, 5.6606e-05, 3.7230e-05, 6.8162e-05])

        SCALE_WEIGHTS = SCALE_WEIGHTS.type_as(pred)
        SCALE_BIASES = SCALE_BIASES.type_as(pred)
        #pred = pred / SCALE_WEIGHTS + SCALE_BIASES

        SCALING_FACTORS = SCALING_FACTORS.type_as(pred)
        '''
        pred = self.softmax(pred)
        #pred = pred * SCALING_FACTORS.unsqueeze(0).unsqueeze(0)  
        #pred = pred / pred.sum(dim=-1, keepdim=True)
            
        
        tokens[:, 0, :] = torch.topk(pred[:, 0, :], beam, dim=1)[1]
        scores[:, :1, :, :] = einops.repeat(pred, "B L V -> B L V S", S=beam)

        # Make all tensors the right shape for decoding.
        precursors = einops.repeat(precursors, "B L -> (B S) L", S=beam)
        mem_masks = einops.repeat(mem_masks, "B L -> (B S) L", S=beam)
        memories = einops.repeat(memories, "B L V -> (B S) L V", S=beam)
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")

        # The main decoding loop.
        for step in range(0, self.max_length):
            # Terminate beams exceeding the precursor m/z tolerance and track
            # all finished beams (either terminated or stop token predicted).
            (
                finished_beams,
                beam_fits_precursor,
                discarded_beams,
            ) = self._finish_beams(tokens, precursors, step)
            # Cache peptide predictions from the finished beams (but not the
            # discarded beams).
            self._cache_finished_beams(
                tokens,
                scores,
                step,
                finished_beams & ~discarded_beams,
                beam_fits_precursor,
                pred_cache,
            )

            # Stop decoding when all current beams have been finished.
            # Continue with beams that have not been finished and not discarded.
            finished_beams |= discarded_beams
            if finished_beams.all():
                break
            # Update the scores.
            
            pred = self.decoder(
                tokens[~finished_beams, : step + 1],
                precursors[~finished_beams, :],
                memories[~finished_beams, :, :],
                mem_masks[~finished_beams, :],
            )
            pred[:, :, discarded] = -10 #torch.min(pred).item()
            #pred = pred / SCALE_WEIGHTS + SCALE_BIASES
            
            pred = self.softmax(pred)
            #pred = pred * SCALING_FACTORS.unsqueeze(0).unsqueeze(0)  
            #pred = pred / pred.sum(dim=-1, keepdim=True)
            
            scores = scores.type_as(pred)
            scores[~finished_beams, : step + 2, :] =  pred
            # Find the top-k beams with the highest scores and continue decoding
            # those.
            tokens, scores = self._get_topk_beams(
                tokens, scores, finished_beams, batch, step + 1
            )

        # Return the peptide with the highest confidence score, within the
        # precursor m/z tolerance if possible.
        return list(self._get_top_peptide(pred_cache))

    def _finish_beams(
        self,
        tokens: torch.Tensor,
        precursors: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Track all beams that have been finished, either by predicting the stop
        token or because they were terminated due to exceeding the precursor
        m/z tolerance.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        step : int
            Index of the current decoding step.

        Returns
        -------
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams have been
            finished.
        beam_fits_precursor: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating if current beams are within precursor m/z
            tolerance.
        discarded_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams should be
            discarded (e.g. because they were predicted to end but violate the
            minimum peptide length).
        """
        n_term = torch.Tensor( self.tokenizer.n_term ).to(self.decoder.device)
        
        beam_fits_precursor = torch.zeros(
            tokens.shape[0], dtype=torch.bool
        ).to(self.decoder.device)
        # Beams with a stop token predicted in the current step can be finished.
        finished_beams = torch.zeros(tokens.shape[0], dtype=torch.bool).to(
            self.decoder.device
        )
        ends_stop_token = tokens[:, step] == self.stop_token
        finished_beams[ends_stop_token] = True
        # Beams with a dummy token predicted in the current step can be
        # discarded.
        discarded_beams = torch.zeros(tokens.shape[0], dtype=torch.bool).to(
            self.decoder.device
        )
        discarded_beams[tokens[:, step] == 0] = True
        # Discard beams with invalid modification combinations (i.e. N-terminal
        # modifications occur multiple times or in internal positions).
        if step > 1:  # Only relevant for longer predictions.
            dim0 = torch.arange(tokens.shape[0])
            final_pos = torch.full((ends_stop_token.shape[0],), step)
            final_pos[ends_stop_token] = step - 1
            # Multiple N-terminal modifications.
            multiple_mods = torch.isin(
                tokens[dim0, final_pos], n_term
            ) & torch.isin(tokens[dim0, final_pos - 1], n_term)
            # N-terminal modifications occur at an internal position.
            # Broadcasting trick to create a two-dimensional mask.
            mask = (final_pos - 1)[:, None] >= torch.arange(tokens.shape[1])
            internal_mods = torch.isin(
                torch.where(mask.to(self.decoder.device), tokens, 0), n_term
            ).any(dim=1)
            discarded_beams[multiple_mods | internal_mods] = True

        # Check which beams should be terminated or discarded based on the
        # predicted peptide.
        for i in range(len(finished_beams)):
            # Skip already discarded beams.
            if discarded_beams[i]:
                continue
            pred_tokens = tokens[i][: step + 1]
            peptide_len = len(pred_tokens)
            
            # Omit stop token.
            if self.tokenizer.reverse and pred_tokens[0] == self.stop_token:
                pred_tokens = pred_tokens[1:]
                peptide_len -= 1
            elif not self.tokenizer.reverse and pred_tokens[-1] == self.stop_token:
                pred_tokens = pred_tokens[:-1]
                peptide_len -= 1
            # Discard beams that were predicted to end but don't fit the minimum
            # peptide length.
            if finished_beams[i] and peptide_len < self.min_peptide_len:
                discarded_beams[i] = True
                continue
            # Terminate the beam if it has not been finished by the model but
            # the peptide mass exceeds the precursor m/z to an extent that it
            # cannot be corrected anymore by a subsequently predicted AA with
            # negative mass.
            precursor_charge = precursors[i, 1]
            precursor_mz = precursors[i, 2]
            matches_precursor_mz = exceeds_precursor_mz = False
            for aa in [None] if finished_beams[i] else self.tokenizer.aa_neg_mass_idx:
                if aa is None:
                    calc_peptide = pred_tokens
                else:
                    calc_peptide = pred_tokens.detach().clone() #.copy()
                    calc_peptide = torch.cat((calc_peptide, torch.tensor([aa]).type_as(calc_peptide)))
                try:
                    calc_mz = self.tokenizer.calc_mass(
                        seq=calc_peptide, charge=precursor_charge
                    )
                    delta_mass_ppm = [
                        _calc_mass_error(
                            calc_mz,
                            precursor_mz,
                            precursor_charge,
                            isotope,
                        )
                        for isotope in range(
                            self.isotope_error_range[0],
                            self.isotope_error_range[1] + 1,
                        )
                    ]
                    # Terminate the beam if the calculated m/z for the predicted
                    # peptide (without potential additional AAs with negative
                    # mass) is within the precursor m/z tolerance.
                    matches_precursor_mz = aa is None and any(
                        abs(d) < self.precursor_mass_tol
                        for d in delta_mass_ppm
                    )
                    # Terminate the beam if the calculated m/z exceeds the
                    # precursor m/z + tolerance and hasn't been corrected by a
                    # subsequently predicted AA with negative mass.
                    if matches_precursor_mz:
                        exceeds_precursor_mz = False
                    else:
                        exceeds_precursor_mz = all(
                            d > self.precursor_mass_tol for d in delta_mass_ppm
                        )
                        exceeds_precursor_mz = (
                            finished_beams[i] or aa is not None
                        ) and exceeds_precursor_mz
                    if matches_precursor_mz or exceeds_precursor_mz:
                        break
                except KeyError:
                    matches_precursor_mz = exceeds_precursor_mz = False
            # Finish beams that fit or exceed the precursor m/z.
            # Don't finish beams that don't include a stop token if they don't
            # exceed the precursor m/z tolerance yet.
            if finished_beams[i]:
                beam_fits_precursor[i] = matches_precursor_mz
            elif exceeds_precursor_mz:
                finished_beams[i] = True
                beam_fits_precursor[i] = matches_precursor_mz
        return finished_beams, beam_fits_precursor, discarded_beams

    def _cache_finished_beams(
        self,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        step: int,
        beams_to_cache: torch.Tensor,
        beam_fits_precursor: torch.Tensor,
        pred_cache: Dict[int, List[Tuple[float, np.ndarray, torch.Tensor]]],
    ):
        """
        Cache terminated beams.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        step : int
            Index of the current decoding step.
        beams_to_cache : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are ready for
            caching.
        beam_fits_precursor: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the beams are within the
            precursor m/z tolerance.
        pred_cache : Dict[int, List[Tuple[float, np.ndarray, torch.Tensor]]]
            Priority queue with finished beams for each spectrum, ordered by
            peptide score. For each finished beam, a tuple with the (negated)
            peptide score, amino acid-level scores, and the predicted tokens is
            stored.
        """
        for i in range(len(beams_to_cache)):
            if not beams_to_cache[i]:
                continue
            # Find the starting index of the spectrum.
            spec_idx = i // self.n_beams
            # FIXME: The next 3 lines are very similar as what's done in
            #  _finish_beams. Avoid code duplication?
            pred_tokens = tokens[i][: step + 1]
            # Omit the stop token from the peptide sequence (if predicted).
            has_stop_token = pred_tokens[-1] == self.stop_token
            pred_peptide = pred_tokens[:-1] if has_stop_token else pred_tokens
            # Don't cache this peptide if it was already predicted previously.
            if any(
                torch.equal(pred_cached[-1], pred_peptide)
                for pred_cached in pred_cache[spec_idx]
            ):
                # TODO: Add duplicate predictions with their highest score.
                continue
            #scores[i : i + 1, : step + 1, :]
            #smx = self.softmax(scores[i : i + 1, : step + 1, :])
            smx = scores[i : i + 1, : step + 1, :]
            prev_smx = scores[i : i + 1, : step + 1, :]
            aa_scores = smx[0, range(len(pred_tokens)), pred_tokens].tolist()
            # Add an explicit score 0 for the missing stop token in case this
            # was not predicted (i.e. early stopping).
            if not has_stop_token:
                aa_scores.append(0)
            aa_scores = np.asarray(aa_scores)
            
            #discarded = torch.tensor(self.tokenizer.discarded_residue_predictions).type_as(pred_peptide)
            #mask = torch.isin(pred_peptide, discarded).cpu().numpy()
            #aa_scores[:-1][mask] = 0
            
            # Calculate the updated amino acid level and the peptide scores.
            aa_scores, peptide_score = _aa_pep_score(
                aa_scores, beam_fits_precursor[i]
            )
            # Omit the stop token from the amino acid-level scores.
            aa_scores = aa_scores[:-1]
            # Add the prediction to the cache (minimum priority queue, maximum
            # the number of beams elements).
            if len(pred_cache[spec_idx]) < self.n_beams:
                heapadd = heapq.heappush
            else:
                heapadd = heapq.heappushpop
                
            _rand = torch.rand(1).item()            
            try:
                heapadd(pred_cache[spec_idx], (peptide_score, 
                                               _rand,  ## add this to avoid duplicated pep scores and aa_scores
                                               aa_scores,                                            
                                               torch.clone(pred_peptide),
                                               prev_smx ),
                       )
            except:
                None
                

    def _get_topk_beams(
        self,
        tokens: torch.tensor,
        scores: torch.tensor,
        finished_beams: torch.tensor,
        batch: int,
        step: int,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Find the top-k beams with the highest scores and continue decoding
        those.

        Stop decoding for beams that have been finished.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are ready for
            caching.
        batch: int
            Number of spectra in the batch.
        step : int
            Index of the next decoding step.

        Returns
        -------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        """
        beam = self.n_beams  # S
        vocab = self.vocab_size + 1  # V

        # Reshape to group by spectrum (B for "batch").
        tokens = einops.rearrange(tokens, "(B S) L -> B L S", S=beam)
        scores = einops.rearrange(scores, "(B S) L V -> B L V S", S=beam)

        # Get the previous tokens and scores.
        prev_tokens = einops.repeat(
            tokens[:, :step, :], "B L S -> B L V S", V=vocab
        )
        prev_scores = torch.gather(
            scores[:, :step, :, :], dim=2, index=prev_tokens
        )
        prev_scores = einops.repeat(
            prev_scores[:, :, 0, :], "B L S -> B L (V S)", V=vocab
        )

        # Get the scores for all possible beams at this step.
        step_scores = torch.zeros(batch, step + 1, beam * vocab).type_as(
            scores
        )
        step_scores[:, :step, :] = prev_scores
        step_scores[:, step, :] = einops.rearrange(
            scores[:, step, :, :], "B V S -> B (V S)"
        )

        # Find all still active beams by masking out terminated beams.
        active_mask = (
            ~finished_beams.reshape(batch, beam).repeat(1, vocab)
        ).float()
        # Mask out the index '0', i.e. padding token, by default.
        # FIXME: Set this to a very small, yet non-zero value, to only
        # get padding after stop token.
        active_mask[:, :beam] = 1e-8

        # Figure out the top K decodings.
        _, top_idx = torch.topk(step_scores.nanmean(dim=1) * active_mask, beam)
        v_idx, s_idx = np.unravel_index(top_idx.cpu(), (vocab, beam))
        s_idx = einops.rearrange(s_idx, "B S -> (B S)")
        b_idx = einops.repeat(torch.arange(batch), "B -> (B S)", S=beam)

        # Record the top K decodings.
        tokens[:, :step, :] = einops.rearrange(
            prev_tokens[b_idx, :, 0, s_idx], "(B S) L -> B L S", S=beam
        )
        tokens[:, step, :] = torch.tensor(v_idx)
        scores[:, : step + 1, :, :] = einops.rearrange(
            scores[b_idx, : step + 1, :, s_idx], "(B S) L V -> B L V S", S=beam
        )
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        return tokens, scores

    def _get_top_peptide(
        self,
        pred_cache: Dict[int, List[Tuple[float, np.ndarray, torch.Tensor]]],
    ) -> Iterable[List[Tuple[float, np.ndarray, str]]]:
        """
        Return the peptide with the highest confidence score for each spectrum.

        Parameters
        ----------
        pred_cache : Dict[int, List[Tuple[float, np.ndarray, torch.Tensor]]]
            Priority queue with finished beams for each spectrum, ordered by
            peptide score. For each finished beam, a tuple with the peptide
            score, amino acid-level scores, and the predicted tokens is stored.

        Returns
        -------
        pred_peptides : Iterable[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide prediction(s). A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        for peptides in pred_cache.values():
            if len(peptides) > 0:
                yield [
                    (
                        pep_score,
                        aa_scores,
                        self.tokenizer.detokenize(pred_tokens.unsqueeze(0))[0],
                        smx
                    )
                    for pep_score, _, aa_scores, pred_tokens, smx in heapq.nlargest(
                        self.top_match, peptides
                    )
                ]
            else:
                yield []

    def _forward_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward learning step.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.
        sequences : List[str] of length n_spectra
            The partial peptide sequences to predict.

        Returns
        -------
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        tokens : torch.Tensor of shape (n_spectra, length)
            The predicted tokens for each spectrum.
        """
        #tokens = self.tokenizer.tokenize(sequences)
        #tokens = tokens.type_as(spectra) #move to correct devide
        #tokens = tokens.type(torch.long) # cast as long
        
        spectra, precursors, tokens = self._process_batch(batch)
        decoded = self.decoder(tokens, precursors, *self.encoder(spectra))
        return decoded, tokens

    def training_step(self, batch, *args, mode: str = "train",) -> torch.Tensor:
        """
        A single training step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences as torch Tensors.
        mode : str
            Logging key to describe the current stage.

        Returns
        -------
        torch.Tensor
            The loss of the training step.
        """
        
        pred, truth = self._forward_step(batch)
        pred_tokens = torch.argmax(pred[:, :-1, :], dim=2)
        pred = pred[:, :-1, :].reshape(-1, self.vocab_size + 1)
        batch_size = truth.shape[0]
        
        if mode == "train":
            loss = self.celoss(pred, truth.flatten())
        else:
            loss = self.val_celoss(pred, truth.flatten())
        self.log(
            f"{mode}_CELoss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size
        )
        return loss

    def validation_step( self, batch, *args ) -> torch.Tensor:
        """
        A single validation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences.

        Returns
        -------
        torch.Tensor
            The loss of the validation step.
        """
    
        # Record the loss.
        loss = self.training_step(batch, mode="valid")
        if not self.calculate_precision:
            return loss

        # Calculate and log amino acid and peptide match evaluation metrics from
        # the predicted peptides.
        peptides_true = self.tokenizer.detokenize(batch['seq'])
        peptides_pred = []
        for spectrum_preds in self.forward(batch):
            for _, _, pred, smx in spectrum_preds:
                peptides_pred.append(pred)
        
        batch_size = len(peptides_true)
        aa_precision, _, pep_precision = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(
                peptides_true,
                peptides_pred,
                self.tokenizer.residues,
            )
        )
        log_args = dict(on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "pep_precision",
            pep_precision,
            **log_args,
            batch_size=batch_size
        )
        self.log(
            "aa_precision",
            aa_precision,
            **log_args,
            batch_size=batch_size
        )
        return loss

    def predict_step( self, batch, *args
                    ) -> List[Tuple[np.ndarray, float, float, str, float, np.ndarray]]:
        """
        A single prediction step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            spectrum identifiers as torch Tensors.

        Returns
        -------
        predictions: List[Tuple[np.ndarray, float, float, str, float, np.ndarray]]
            Model predictions for the given batch of spectra containing spectrum
            ids, precursor information, peptide sequences as well as peptide
            and amino acid-level confidence scores.
        """
        
        _, precursors, true_seqs = self._process_batch(batch)
        scans, titles, file_names = batch["scans"], batch["title"], batch["peak_file"]
        
        true_seqs = (self.tokenizer.detokenize(true_seqs, join=True, trim_stop_token=True) 
                             if true_seqs is not None else ['']*len(scans)  )

        prec_charges = precursors[:, 1].cpu().detach().numpy()
        prec_mzs = precursors[:, 2].cpu().detach().numpy()
            
        predictions = []
        all_smx = []
        for (
            precursor_charge,
            precursor_mz,
            scan,
            title,
            file_name,
            true_seq,
            spectrum_preds,
            
            
        ) in zip(
            prec_charges,
            prec_mzs,
            scans,
            titles,
            file_names,
            true_seqs,
            self.forward(batch),
            
        ):
            for peptide_score, aa_scores, peptide, smx in spectrum_preds:
                predictions.append(
                    (
                        scan,
                        precursor_charge,
                        precursor_mz,
                        peptide,
                        peptide_score,
                        aa_scores,
                        file_name,
                        true_seq,
                        title, 
                        smx
                    )
                )
                all_smx.append(smx)
        max_pep_len = max(tensor.size(1) for tensor in all_smx)
        smx_padded = torch.vstack([torch.nn.functional.pad(tensor, 
                                       pad=(0, 0,  0, max_pep_len - tensor.size(1), 0, 0),
                                       value=-1
                                      ) for tensor in all_smx]
                         )

        return predictions#, smx_padded
    
    def on_train_start(self):
        # Optimizer settings.
        self.log("hp/optimizer_warmup_iters", self.warmup_iters)
        self.log("hp/optimizer_max_iters", self.max_iters)
        self.log("hp/optimizer_start_lr", self.lr)
        self.log("hp/optimizer_weight_decay", self.weight_decay)
    
    def on_train_epoch_end(self) -> None:
        """
        Log the training loss at the end of each epoch.
        """
        if "train_CELoss" in self.trainer.callback_metrics:
            train_loss = self.trainer.callback_metrics["train_CELoss"].detach().item()
        else:
            train_loss = np.nan
        metrics = {
            "step": self.trainer.global_step,
            "train": train_loss,
        }
        self._history.append(metrics)
        self._log_history()

    def on_validation_epoch_end(self) -> None:
        """
        Log the validation metrics at the end of each epoch.
        """
        callback_metrics = self.trainer.callback_metrics
        metrics = {
            "step": self.trainer.global_step,
            "valid": callback_metrics["valid_CELoss"].detach().item(),
        }

        if self.calculate_precision:
            metrics["valid_aa_precision"] = (
                callback_metrics["aa_precision"].detach().item()
            )
            metrics["valid_pep_precision"] = (
                callback_metrics["pep_precision"]
                .detach()
                .item()
            )
        self._history.append(metrics)
        self._log_history()

    def on_predict_batch_end(
        self,
        outputs: List[Tuple[np.ndarray, List[str], torch.Tensor]],
        *args,
    ) -> None:
        """
        Write the predicted peptide sequences and amino acid scores to the
        output file.
        """
        if self.out_writer is None:
            return
        
        # Triply nested lists: results -> batch -> step -> spectrum.
        for (
            scan,
            charge,
            precursor_mz,
            peptide,
            peptide_score,
            aa_scores,
            file_name,
            true_seq,
            title,
            smx
        ) in outputs:
            if len(peptide) == 0:
                continue
            ### not efficient... compute mass before detokenizing
            tokenized_pep = self.tokenizer.tokenize(peptide, to_strings=True)[0]
            calc_mass = self.tokenizer.calc_mass(tokenized_pep, charge)
            self.out_writer.psms.append(
                (
                    peptide,
                    scan,
                    peptide_score,
                    charge,
                    precursor_mz,
                    calc_mass,
                    ",".join(list(map("{:.5f}".format, aa_scores))),
                    file_name,
                    true_seq,
                    title, 
                ),
            )
                   

    def _log_history(self) -> None:
        """
        Write log to console, if requested.
        """
        # Log only if all output for the current epoch is recorded.
        if len(self._history) == 0:
            return
        if len(self._history) == 1:
            header = "Step\tTrain loss\tValid loss\t"
            if self.calculate_precision:
                header += "Peptide precision\tAA precision"

            logger.info(header)
        metrics = self._history[-1]
        if metrics["step"] % self.n_log == 0:
            msg = "%i\t%.6f\t%.6f"
            vals = [
                metrics["step"],
                metrics.get("train", np.nan),
                metrics.get("valid", np.nan),
            ]

            if self.calculate_precision:
                msg += "\t%.6f\t%.6f"
                vals += [
                    metrics.get("valid_pep_precision", np.nan),
                    metrics.get("valid_aa_precision", np.nan),
                ]

            logger.info(msg, *vals)
            
    def configure_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
        """
        Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized Adam optimizer and its learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                     **self.opt_kwargs)
        # Apply learning rate scheduler per step.
        lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.warmup_iters, max_iters=self.max_iters
        )
        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup : int
        The number of warm up iterations.
    max_iters : torch.optim
        The total number of iterations.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, warmup: int, max_iters: int
    ):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch / self.warmup
        return lr_factor


def _calc_mass_error(
    calc_mz: float, obs_mz: float, charge: int, isotope: int = 0
) -> float:
    """
    Calculate the mass error in ppm between the theoretical m/z and the observed
    m/z, optionally accounting for an isotopologue mismatch.

    Parameters
    ----------
    calc_mz : float
        The theoretical m/z.
    obs_mz : float
        The observed m/z.
    charge : int
        The charge.
    isotope : int
        Correct for the given number of C13 isotopes (default: 0).

    Returns
    -------
    float
        The mass error in ppm.
    """
    return (calc_mz - (obs_mz - isotope * 1.00335 / charge)) / obs_mz * 10**6


def _aa_pep_score(
    aa_scores: np.ndarray, fits_precursor_mz: bool
) -> Tuple[np.ndarray, float]:
    """
    Calculate amino acid and peptide-level confidence score from the raw amino
    acid scores.

    The peptide score is the mean of the raw amino acid scores. The amino acid
    scores are the mean of the raw amino acid scores and the peptide score.

    Parameters
    ----------
    aa_scores : np.ndarray
        Amino acid level confidence scores.
    fits_precursor_mz : bool
        Flag indicating whether the prediction fits the precursor m/z filter.

    Returns
    -------
    aa_scores : np.ndarray
        The amino acid scores.
    peptide_score : float
        The peptide score.
    """
    peptide_score = np.mean(aa_scores)
    aa_scores = (aa_scores + peptide_score) / 2
    if not fits_precursor_mz:
        peptide_score -= 1
    return aa_scores, peptide_score
