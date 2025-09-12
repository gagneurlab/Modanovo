"""Training and testing functionality for the de novo peptide sequencing
model."""

import glob
import logging
import os
import tempfile
import uuid
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Union

import lightning.pytorch as pl
import numpy as np
import torch
from datetime import datetime

from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from ..config import Config
from ..data import ms_io
from ..denovo.dataloaders import DeNovoDataModule
from ..denovo.model import Spec2Pep
from . import DanielasPeptideTokenizer



logger = logging.getLogger("modanovo")


class ModelRunner:
    """A class to run models.

    Parameters
    ----------
    config : Config object
        The configuration.
    model_filename : str, optional
        The model filename is required for eval and de novo modes,
        but not for training a model from scratch.
    """

    def __init__(
        self,
        config: Config,
        model_filename: Optional[str] = None,
    ) -> None:
        """Initialize a ModelRunner"""
        self.config = config
        self.model_filename = model_filename

        # Initialized later:
        self.tmp_dir = None
        self.trainer = None
        self.model = None
        self.loaders = None
        self.writer = None

        # Configure checkpoints.
        self.callbacks = []
        print("CONFIG", config.save_top_k)
        if config.save_top_k is not None:
            self.callbacks = [
                ModelCheckpoint(
                    dirpath=config.model_save_folder_path,
                    monitor="valid_CELoss",
                    mode="min",
                    save_top_k=config.save_top_k,
                    #save_weights_only=config.save_weights_only,
                    auto_insert_metric_name=True,
                    filename='{epoch}-{step}-{train_CELoss:.3f}-{valid_CELoss:.3f}',
                    save_last=True,
                )
            ]
            
        if config.early_stopping_patience is not None:
            self.callbacks.append( EarlyStopping(
                     monitor="valid_CELoss", 
                     min_delta=0.00, 
                     patience=self.config.early_stopping_patience, 
                     verbose=True, 
                     check_finite=True,
                     mode="min")
                                 )
        if config.tb_summarywriter is not None: 
            self.callbacks.append(LearningRateMonitor(logging_interval='step', log_momentum=True))
        
        
    def __enter__(self):
        """Enter the context manager"""
        self.tmp_dir = tempfile.TemporaryDirectory()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Cleanup on exit"""
        self.tmp_dir.cleanup()
        self.tmp_dir = None
        if self.writer is not None:
            self.writer.save()

    def train(
        self,
        train_peak_path: Iterable[str],
        valid_peak_path: Iterable[str],
    ) -> None:
        """Train the model.

        Parameters
        ----------
        train_peak_path : iterable of str
            The path to the MS data files for training.
        valid_peak_path : iterable of str
            The path to the MS data files for validation.

        Returns
        -------
        self
        """
        logger.info('=== CONFIG ===')
        logger.info(self.config.items())
        logger.info('===============')
        
        self.initialize_tokenizer()
        self.initialize_trainer(train=True)
        
        self.initialize_data_module(train_peak_path, valid_peak_path)
        
        logger.info(f'Finished building data loaders' ) #, n_workers: {self.config.n_workers}
        _train_loader = self.train_loader.train_dataloader()
        _val_loader = self.val_loader.val_dataloader()
        
        logger.info(f'Train batch size {next(iter(_train_loader))["precursor_mz"].shape}' )
        logger.info(f'Val batch size {next(iter(_val_loader))["precursor_mz"].shape }' )
        
        
        
        warmup_iters= (round(int(len(_train_loader)
                           /(torch.cuda.device_count()*self.config.accumulate_grad_batches)) 
                       *  self.config.warm_up_epochs )
                      )
        warmup_iters = max(warmup_iters,1 )
        max_iters = (int(len(_train_loader)
                         /(torch.cuda.device_count()*self.config.accumulate_grad_batches))
                     * int(self.config.max_epochs) 
                    )
        max_iters = max(max_iters, 1)
         
        self.initialize_model(train=True, warmup_iters=warmup_iters, max_iters=max_iters)

        self.trainer.fit(self.model, _train_loader, _val_loader, 
                         ckpt_path=self.config.resume_training_from)


    def evaluate(self, peak_path: Iterable[str]) -> None:
        """Evaluate peptide sequence preditions from a trained model.

        Parameters
        ----------
        peak_path : iterable of str
            The path with MS data files for predicting peptide sequences.

        Returns
        -------
        self
        """
        self.initialize_tokenizer()
        self.initialize_trainer(train=False)
        self.initialize_model(train=False)

        self.initialize_data_module(peak_path=peak_path, train=False, annotated=True)
        
        self.trainer.validate(self.model, self.test_loader.test_dataloader())

    def predict(self, peak_path: Iterable[str], output: str) -> None:
        """Predict peptide sequences with a trained model.

        Parameters
        ----------
        peak_path : iterable of str
            The path with the MS data files for predicting peptide sequences.
        output : str
            Where should the output be saved?

        Returns
        -------
        self
        """
        self.writer = ms_io.MztabWriter(Path(output).with_suffix(".mztab"))
        self.writer.set_metadata(
            self.config,
            model=str(self.model_filename),
            config_filename=self.config.file,
        )

        self.initialize_tokenizer()
        self.initialize_trainer(train=False)
        self.initialize_model(train=False)
        self.model.out_writer = self.writer

        self.writer.set_ms_run(peak_path)
        self.initialize_data_module(peak_path=peak_path, train=False, 
                                    annotated=self.config.save_correct_seq)
        
        self.trainer.predict(self.model, self.test_loader.test_dataloader())

    def initialize_tokenizer(self) -> None:
        
        self.tokenizer = DanielasPeptideTokenizer(
            expanded_residues=self.config.expanded_residues,
            replace_isoleucine_with_leucine=True,
            reverse=True,
            discarded_residue_predictions=self.config.discarded_residue_predictions
        )
        
        logger.info('====== TOKENIZER ========')
        logger.info(f"\t RESIDUES {self.tokenizer.residues}")
        logger.info(f'\t n_default_residues {self.tokenizer.n_default_residues}')
        logger.info(f'\t n_expanded_residues {self.tokenizer.n_expanded_residues}')
        logger.info(f'\t len(tokenizer) {len(self.tokenizer)}')
        logger.info(f'\t discarded_residue_predictions {self.tokenizer.discarded_residue_predictions}')
  
    def initialize_trainer(self, train: bool) -> None:
        """Initialize the lightning Trainer.

        Parameters
        ----------
        train : bool
            Determines whether to set the trainer up for model training
            or evaluation / inference.
        """
        trainer_cfg = dict(
            accelerator=self.config.accelerator,
            devices=1,
            enable_checkpointing=False,
            precision=self.config.precision,
            logger = False
        )

        if train:
            if self.config.devices is None:
                devices = "auto"
            else:
                devices = self.config.devices
 
            if self.config.tb_summarywriter is not None:
                logger = TensorBoardLogger(self.config.tb_summarywriter, 
                                       version=None,
                                       name=f'model_{datetime.now() .strftime("%Y%m%d_%H%M")}',
                                      default_hp_metric=False)
            else:
                logger = False

            additional_cfg = dict(
                devices=devices,
                callbacks=self.callbacks,
                enable_checkpointing=self.config.save_top_k is not None,
                max_epochs=self.config.max_epochs,
                num_sanity_val_steps=self.config.num_sanity_val_steps,
                strategy=self._get_strategy(),
                val_check_interval=self.config.val_check_interval,
                check_val_every_n_epoch=None,
                logger=logger,
                accumulate_grad_batches=self.config.accumulate_grad_batches,
                gradient_clip_val=self.config.gradient_clip_val,
                gradient_clip_algorithm=self.config.gradient_clip_algorithm,
        

            )
            trainer_cfg.update(additional_cfg)

        self.trainer = pl.Trainer(**trainer_cfg)

    def initialize_model(self, train: bool, warmup_iters=0, max_iters=0) -> None:
        """Initialize the model.

        Parameters
        ----------
        train : bool
            Determines whether to set the model up for model training
            or evaluation / inference.
        """
        try:
            tokenizer = self.tokenizer
        except AttributeError:
            raise RuntimeError("Please use `initialize_tokenizer()` first.")
    
        model_params = dict(
            dim_model=self.config.dim_model,
            n_head=self.config.n_head,
            dim_feedforward=self.config.dim_feedforward,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
            dim_intensity=self.config.dim_intensity,
            max_length=self.config.max_length,
            max_charge=self.config.max_charge,
            precursor_mass_tol=self.config.precursor_mass_tol,
            isotope_error_range=self.config.isotope_error_range,
            min_peptide_len=self.config.min_peptide_len,
            n_beams=self.config.n_beams,
            top_match=self.config.top_match,
            n_log=self.config.n_log,
            train_label_smoothing=self.config.train_label_smoothing,
            warmup_iters=warmup_iters,
            max_iters=max_iters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            out_writer=self.writer,
            calculate_precision=self.config.calculate_precision,
            tokenizer=tokenizer,
            train_from_default_residues=self.config.train_from_default_residues if train else False
        )

        if self.model_filename is None:
            # Train a model from scratch if no model file is provided.
            if train:
                self.model = Spec2Pep(**model_params)
                return
            # Else we're not training, so a model file must be provided.
            else:
                logger.error("A model file must be provided")
                raise ValueError("A model file must be provided")
        # Else a model file is provided (to continue training or for inference).

        if not Path(self.model_filename).exists():
            logger.error(
                "Could not find the model weights at file %s",
                self.model_filename,
            )
            raise FileNotFoundError("Could not find the model weights file")

            
            
        # This only doesn't work if the weights are from an older version
        try:
            print('WEIGHTS ONLY')
            self.model = Spec2Pep.load_from_checkpoint(
                self.model_filename,
                strict=False,
                 #weights_only=True,   
               # map_location=torch.empty(1).device ,
                **model_params,
            )
        except RuntimeError:
            raise RuntimeError(
                "Weights file incompatible with the current version."
            )

    def initialize_data_module(
        self, peak_path, peak_path_val=None, train=True, annotated=True,
    ) -> None:
        """Initialize the data module

        Parameters
        ----------
        """
        try:
            n_devices = self.trainer.num_devices
            train_bs = self.config.train_batch_size // n_devices
            eval_bs = self.config.predict_batch_size // n_devices
        except AttributeError:
            raise RuntimeError("Please use `initialize_trainer()` first.")
        
        try:
            tokenizer = self.tokenizer
        except AttributeError:
            raise RuntimeError("Please use `initialize_tokenizer()` first.")
        
        

        dataloader_params = dict( n_workers=0,#self.config.n_workers,
                                  train_batch_size=train_bs,
                                  eval_batch_size=eval_bs,
                                  min_mz=self.config.min_mz,
                                  max_mz=self.config.max_mz,
                                  min_intensity=self.config.min_intensity,
                                  remove_precursor_tol=self.config.remove_precursor_tol,
                                  lance_dir_name=self.config.lance_dir_name,
                                  n_peaks=self.config.n_peaks,
                                  max_charge=self.config.max_charge,
                                  tokenizer=tokenizer,
                                  shuffle=self.config.shuffle,
                                  buffer_size=self.config.buffer_size,
                                )
                          
        if not train:
            self.test_loader = DeNovoDataModule( test_dataset_paths=peak_path, **dataloader_params )
            self.test_loader.setup(stage="test", annotated=annotated)

        else:
            self.train_loader = DeNovoDataModule(train_dataset_paths=peak_path, **dataloader_params    )
            self.train_loader.setup()

            self.val_loader = DeNovoDataModule(valid_dataset_paths=peak_path_val, **dataloader_params)
            self.val_loader.setup()

    def _get_strategy(self) -> Union[str, DDPStrategy]:
        """Get the strategy for the Trainer.

        The DDP strategy works best when multiple GPUs are used. It can work
        for CPU-only, but definitely fails using MPS (the Apple Silicon chip)
        due to Gloo.

        Returns
        -------
        Union[str, DDPStrategy]
            The strategy parameter for the Trainer.

        """
        if self.config.accelerator in ("cpu", "mps"):
            return "auto"
        elif self.config.devices == 1:
            return "auto"
        elif torch.cuda.device_count() > 1:
            return DDPStrategy(find_unused_parameters=False, static_graph=True)
        else:
            return "auto"


