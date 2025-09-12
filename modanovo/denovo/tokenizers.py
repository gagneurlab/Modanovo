"""Tokenizers for peptides."""
from __future__ import annotations

import re
from collections.abc import Iterable
from collections import OrderedDict

import pprint
import numba as nb
import numpy as np
import torch
from pyteomics.proforma import GenericModification, MassModification

from . import Peptide

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import torch
from sortedcontainers import SortedDict, SortedSet
from torch import nn


from typing import Any
import polars as pl


def listify(obj: Any) -> list[Any]:  # noqa: ANN401
    """Turn an object into a list, but don't split strings."""
    try:
        invalid = [str, pl.DataFrame, pl.LazyFrame]
        if any(isinstance(obj, c) for c in invalid):
            raise TypeError

        iter(obj)
    except (AssertionError, TypeError):
        obj = [obj]

    return list(obj)


class BaseTokenizer(ABC):
    """An abstract base class for Depthcharge tokenizers.

    Parameters
    ----------
    tokens : Sequence[str]
        The tokens to consider.
    stop_token : str, optional
        The stop token to use.
    add_stop : bool, optional
            Append the stop token tothe end of the sequence.

    """

    def __init__(self, tokens: Sequence[str], 
                 stop_token: str = "$", add_stop: bool = True,
                reverse:bool = True
                
                ) -> None:
        """Initialize a tokenizer."""
        self.stop_token = stop_token
        self.add_stop = add_stop
        self.reverse = reverse

        if not self.stop_token in tokens:
            tokens.add(self.stop_token)
       # else:
       #     print(f"Stop token {stop_token} already exists in tokens.")
            
        self.index = {k: i + 1 for i, k in enumerate(tokens)}
        pprint.pprint(self.index)
        self.reverse_index = [None] + list(tokens)
        self.stop_int = self.index[self.stop_token]

    def __len__(self) -> int:
        """The number of tokens."""
        return len(self.index)

    @abstractmethod
    def split(self, sequence: str) -> list[str]:
        """Split a sequence into the constituent string tokens."""

    def tokenize(
        self,
        sequences: Iterable[str],
        to_strings: bool = False,
    ) -> torch.Tensor | list[list[str]]:
        """Tokenize the input sequences.

        Parameters
        ----------
        sequences : Iterable[str]
            The sequences to tokenize.
        to_strings : bool, optional
            Return each as a list of token strings rather than a
            tensor. This is useful for debugging.
        
        Returns
        -------
        torch.Tensor of shape (n_sequences, max_length) or list[list[str]]
            Either a tensor containing the integer values for each
            token, padded with 0's, or the list of tokens comprising
            each sequence.
        """
        try:
            out = []
            for seq in listify(sequences):
                tokens = self.split(seq)
                if self.add_stop and tokens[-1] != self.stop_token:
                    tokens.append(self.stop_token)

                if to_strings:
                    out.append(tokens)
                    continue

                out.append(torch.tensor([self.index[t] for t in tokens]))

            if to_strings:
                return out

            if isinstance(sequences, str):
                return out[0]

            return nn.utils.rnn.pad_sequence(out, batch_first=True)
        except KeyError as err:
            raise ValueError("Unrecognized token") from err
            #return utils.listify(sequences) # Useful for prediction mode when residues of correct not known 

    def detokenize(
        self,
        tokens: torch.Tensor,
        join: bool = True,
        trim_stop_token: bool = True,
        return_pep_lens: bool = False
    ) -> list[str] | list[list[str]]:
        """Retreive sequences from tokens.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_sequences, max_length)
            The zero-padded tensor of integerized tokens to decode.
        join : bool, optional
            Join tokens into strings?
        trim_stop_token : bool, optional
            Remove the stop token from the end of a sequence.

        Returns
        -------
        list[str] or list[list[str]]
            The decoded sequences each as a string or list or strings.
        """
        decoded = []
        pep_lens = []
        for row in tokens:
            seq = [
                self.reverse_index[i]
                for i in row
                if self.reverse_index[i] is not None
            ]
            
            #if trim_stop_token and seq[-1] == self.stop_token:
            #    seq.pop(-1)
            
            if trim_stop_token and self.stop_token in seq:
                idx = seq.index(self.stop_token)
                seq = seq[: idx]
            #    seq = seq.replace(self.stop_token, '')   

            if self.reverse:
                    seq = seq[::-1]
            
            
            if join:
                seq = "".join(seq)
            
                
            decoded.append(seq)

        return decoded

class DanielasPeptideTokenizer(BaseTokenizer):
    """A tokenizer for ProForma peptide sequences.

    Parse and tokenize ProForma-compliant peptide sequences. Additionally,
    use this class to calculate fragment and precursor ion masses.

    Parameters
    ----------
    residues : dict[str, float], optional
        Residues and modifications to add to the vocabulary beyond the
        standard 20 amino acids.
    replace_isoleucine_with_leucine : bool
        Replace I with L residues, because they are isobaric and often
        indistinguishable by mass spectrometry.
    reverse : bool
        Reverse the sequence for tokenization, C-terminus to N-terminus.

    Attributes
    ----------
    residues : numba.typed.Dict[str, float]
        The residues and modifications and their associated masses.
        terminal modifcations are indicated by `-`.
    index : SortedDict{str, int}
        The mapping of residues and modifications to integer representations.
    reverse_index : list[None | str]
        The ordered residues and modifications where the list index is the
        integer representation for a token.
    stop_token : str
        The stop token.
    """

    residues = OrderedDict({
                      "G": 57.021464,
                      "A": 71.037114,
                      "S": 87.032028,
                      "P": 97.052764,
                      "V": 99.068414,
                      "T": 101.047670,
                      "C[+57.021]": 160.030649, # 103.009185 + 57.021464
                      "L": 113.084064,              
                      "I": 113.084064,
                      "N": 114.042927,
                      "D": 115.026943,
                      "Q": 128.058578,
                      "K": 128.094963,
                      "E": 129.042593,
                      "M": 131.040485,
                      "H": 137.058912,
                      "F": 147.068414,
                      "R": 156.101111,
                      "Y": 163.063329,
                      "W": 186.079313,
        
                      # Amino acid modifications.
                      "M[+15.995]": 147.035400,    # Met oxidation:   131.040485 + 15.994915
                      "N[+0.984]": 115.026943,     # Asn deamidation: 114.042927 +  0.984016
                      "Q[+0.984]": 129.042594,     # Gln deamidation: 128.058578 +  0.984016
                      # N-terminal modifications.
                      "[+42.011]-": 42.010565,      # Acetylation
                      "[+43.006]-": 43.005814,     # Carbamylation
                      "[-17.027]-": -17.026549,     # NH3 loss
                      
                      #"[+43.006-17.027]-": 25.980265,   # Carbamylation and NH3 loss              
                      "[+25.980]-":  25.980265,

                      "$":0, ########### stop / end token
                })
    

    # The peptide parsing function:
    #_parse_peptide = Peptide.from_proforma
    _parse_peptide = Peptide.from_massivekb

    def __init__(
        self,
        expanded_residues: dict[str, float] | None = None,
        replace_isoleucine_with_leucine: bool = True,
        reverse: bool = True, ##### very important here
        discarded_residue_predictions = [],
    ) -> None:
        """Initialize a PeptideTokenizer."""
        self.replace_isoleucine_with_leucine = replace_isoleucine_with_leucine
        self.residues = self.residues.copy()
        
        self.n_default_residues = len(self.residues)
        
        if expanded_residues is not None:
            # remove residues that are already contained in default residues to avoid messing up with the order
            self.expanded_residues = {k:v for k,v in expanded_residues.items() if k not in self.residues.keys()}
            self.residues.update(self.expanded_residues)
            self.n_expanded_residues = len(self.expanded_residues)
        else:
            self.expanded_residues = {}
            self.n_expanded_residues = 0
            
        #if self.replace_isoleucine_with_leucine:
        #    del self.residues["I"]

        
        super().__init__(list(self.residues.keys()), reverse=reverse)
        
        # Constants
        self.hydrogen = 1.007825035
        self.oxygen = 15.99491463
        self.h2o = 2 * self.hydrogen + self.oxygen
        self.proton = 1.00727646688
        
        self.n_term = [self.index[aa] for aa in self.index if aa.startswith(('[+', '[-')) ]
        self.aa_neg_mass = [None]
        self.aa_neg_mass_idx = []
        for aa, mass in self.residues.items():
            if mass < 0:
                self.aa_neg_mass.append(aa)
                self.aa_neg_mass_idx.append(self.index[aa])
        self.discarded_residue_predictions = [self.index[aa] for aa in self.index if aa in discarded_residue_predictions ]

    def split(self, sequence: str) -> list[str]:
        """Split a ProForma peptide sequence.

        Parameters
        ----------
        sequence : str
            The peptide sequence.

        Returns
        -------
        list[str]
            The tokens that comprise the peptide sequence.
        """
        #print(sequence)
        original_sequence = sequence
        #sequence = sequence.replace('[+43.006]--17.027', '[+25.980]-')
        try:
            pep = self._parse_peptide(sequence)
        except:
            print(original_sequence)
            print(sequence)
            
        
        if self.replace_isoleucine_with_leucine:
            pep.sequence = pep.sequence.replace("I", "L")
        pep.sequence = pep.sequence.replace("E[-18.011]","Q[-17.027]")
            
        pep = self._split_peptide(pep)
        if self.reverse:
            pep.reverse()

        return pep
    
    def _split_peptide(self, peptide):
        """Split the modified peptide for tokenization."""
        if peptide.modifications is None:
            return list(peptide.sequence)

        out = []
        for idx, (aa, mods) in enumerate(
            zip(f"-{peptide.sequence}-", peptide.modifications)
        ):
            if mods is None:
                if idx and (idx < len(peptide.modifications) - 1):
                    out.append(aa)

                continue

            if len(mods) == 1:
                #try:
                #    modstr = f"[{mods[0].name}]"
                #except (AttributeError, ValueError):
                modstr = f"[{mods[0].mass:+0.3f}]"
            else:
                modstr = f"[{sum([m.mass for m in mods]):+0.3f}]"
                
            if not idx:
                out.append(f"{modstr}-")
            else:
                out.append(f"{aa}{modstr}")

        return out
    
    def calc_mass(self, seq, charge = None):
        if isinstance(seq[0], str):
            calc_mass = sum([self.residues[aa] for aa in seq]) + self.h2o
        else:
            calc_mass = sum([self.residues[self.reverse_index[aa]] for aa in seq]) + self.h2o
        if charge is not None:
            calc_mass = (calc_mass / charge) + self.proton

        return calc_mass

        