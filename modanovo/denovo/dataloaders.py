"""Data loaders for the de novo sequencing task."""
import os
from typing import List, Optional, Tuple
import logging
import tempfile
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from pathlib import Path
import pyarrow as pa
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

from depthcharge.data import (
                                AnnotatedSpectrumDataset,
                                CustomField,
                                SpectrumDataset,
                                preprocessing
)

logger = logging.getLogger("modanovo")


def scale_to_unit_norm(spectrum):
    """
    Scaling function used in Casanovo
    slightly differing from the depthcharge implementation
    """
    spectrum._inner._intensity = spectrum.intensity / np.linalg.norm(
                spectrum.intensity
            )
    return spectrum


class DeNovoDataModule(pl.LightningDataModule):
    """
    Data loader to prepare MS/MS spectra for a Spec2Pep predictor.

    Parameters
    ----------
    train_index : Optional[str]
        Paths to the training data.
    valid_index : Optional[str]
        Paths to the validation data.
    test_index : Optional[str]
        Paths to the testing data.
    batch_size : int
        The batch size to use for training and evaluating.
    n_peaks : Optional[int]
        The number of top-n most intense peaks to keep in each spectrum. `None`
        retains all peaks.
    min_mz : float
        The minimum m/z to include. The default is 140 m/z, in order to exclude
        TMT and iTRAQ reporter ions.
    max_mz : float
        The maximum m/z to include.
    min_intensity : float
        Remove peaks whose intensity is below `min_intensity` percentage of the
        base peak intensity.
    remove_precursor_tol : float
        Remove peaks within the given mass tolerance in Dalton around the
        precursor mass.
    n_workers : int, optional
        The number of workers to use for data loading. By default, the number of
        available CPU cores on the current machine is used.
    random_state : Optional[int]
        The NumPy random state. ``None`` leaves mass spectra in the order they
        were parsed.
    """

    def __init__(
        self,
        train_dataset_paths: Optional[str] = None,
        valid_dataset_paths: Optional[str] = None,
        test_dataset_paths: Optional[str] = None,
        train_batch_size: int = 128,
        eval_batch_size: int = 128,
        n_peaks: Optional[int] = 150,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        max_charge: int = 10,
        n_workers: Optional[int] = None,
        random_state: Optional[int] = None,
        tokenizer =None,
        lance_dir_name=None,
        buffer_size=100_000,
        shuffle=True
    ):
        super().__init__()
        
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.remove_precursor_tol = remove_precursor_tol
        self.max_charge = max_charge
        self.n_workers = n_workers
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        
        self.lance_dir_name = (tempfile.TemporaryDirectory(suffix='.lance').name 
                              if lance_dir_name is None else lance_dir_name )
        
        logger.info(f'Lance dir name: {self.lance_dir_name}')
        
        
        self.valid_charge = np.arange(1, max_charge+1)
        self.preprocessing_fn = [
                            preprocessing.set_mz_range(min_mz=min_mz, max_mz=max_mz),
                            preprocessing.remove_precursor_peak(remove_precursor_tol, "Da"),
                            preprocessing.filter_intensity(min_intensity, n_peaks),
                            preprocessing.scale_intensity("root",1),
                            scale_to_unit_norm #preprocessing.scale_to_unit_norm,
                            ]
        self.custom_field_unannotated = [ CustomField( "scans", 
                                                    lambda x: x["params"]["scans"] if 'scans' in x["params"] else x["params"]["title"],
                                                    pa.string()),
                                         
                                         CustomField( "title", 
                                                    lambda x: x["params"]["title"], pa.string())
                                        ]
        self.custom_field_annotated = self.custom_field_unannotated + [  CustomField( "seq", lambda x: x["params"]["seq"], pa.string())
                                      ]
                            
        
        
        self.train_dataset_paths = train_dataset_paths
        self.valid_dataset_paths = valid_dataset_paths
        self.test_dataset_paths = test_dataset_paths
        self.tokenizer = tokenizer 
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        
        self.n_workers = n_workers if n_workers is not None else os.cpu_count()
        self.rng = np.random.default_rng(random_state)
        
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def check_existing_lance(self, paths, lance_path, annotated=True):
        ## check if lance dir exists
        custom_field = self.custom_field_annotated if annotated else self.custom_field_unannotated
        try:
            if annotated:
                dset = AnnotatedSpectrumDataset.from_lance(path=lance_path, 
                                                           annotations="seq",
                                                tokenizer=self.tokenizer,
                                                custom_fields=custom_field,
                                                preprocessing_fn=self.preprocessing_fn,
                                                valid_charge=self.valid_charge)
            else:
                dset = SpectrumDataset.from_lance(path=lance_path, 
                                                preprocessing_fn=self.preprocessing_fn,
                                                valid_charge=self.valid_charge,
                                                 custom_fields=custom_field
                                                 )
                
        except ValueError:
            logger.info(f'Dataset .lance at <{lance_path}> does not exist. Creating new .lance dataset')
            return
        
        ## check if all peak files are present
        ## TODO: instead of returning None when not equal add missing peak files
        if set(dset.peak_files) == set([os.path.basename(p) for p in paths]):
            logger.info(f'Using existing lance dataset at <{lance_path}>')
            return dset
        else:
            logger.info('Not all paths are present in the existing .lance dataset. Creating new dataset.')
            logger.info(f'Files in dataset: {set(dset.peak_files)} \nexpected files: {set(paths)}')
            
        return
        
    
    def create_new_lance(self, paths, lance_path, annotated=True, shuffle=False):
        
        ## check if already exists
        custom_field = self.custom_field_annotated if annotated else self.custom_field_unannotated
        dset = self.check_existing_lance(paths, lance_path, annotated)
        if dset is None:
            if annotated:
                ## create new
                dset = AnnotatedSpectrumDataset(spectra=paths[0],
                                                  annotations="seq",
                                                  tokenizer=self.tokenizer,
                                                  custom_fields=custom_field,
                                                  path=lance_path,
                                                  preprocessing_fn=self.preprocessing_fn,
                                                  valid_charge=self.valid_charge,
                        )
            else:
                dset = SpectrumDataset(spectra=paths[0],
                                        path=lance_path,
                                        preprocessing_fn=self.preprocessing_fn,
                                        valid_charge=self.valid_charge,
                                       custom_fields=custom_field
                                    )
    
            for i in range(1, len(paths)):
                if annotated:
                    dset.add_spectra(paths[i],
                                   custom_fields=custom_field,
                                   preprocessing_fn=self.preprocessing_fn,
                                   valid_charge=self.valid_charge,
                                )
                else:
                    dset.add_spectra(paths[i],
                                   preprocessing_fn=self.preprocessing_fn,
                                   valid_charge=self.valid_charge,
                                      custom_fields=custom_field
                                )

        #if shuffle:
            #print('SHUFFLING')
            #dset = ShufflerIterDataPipe(dset, buffer_size=self.buffer_size)
        return dset
                
        
    def setup(self, stage: str = None, annotated: bool = True) -> None:
        """
        Set up the PyTorch Datasets.

        Parameters
        ----------
        stage : str {"fit", "validate", "test"}
            The stage indicating which Datasets to prepare. All are prepared by
            default.
        annotated: bool
            True if peptide sequence annotations are available for the test
            data.
        """
        folder, name = os.path.split(self.lance_dir_name)
        lance_train_path = Path(folder, f'train__{name}')
        lance_valid_path = Path(folder, f'val__{name}')
        lance_test_path = Path(folder, f'test__{name}')
        
        if stage in (None, "fit", "validate"):
            
            
            
            if self.train_dataset_paths is not None:
                self.train_dataset = self.create_new_lance( self.train_dataset_paths,
                                                           lance_train_path, shuffle=self.shuffle )
                logger.info(f'LANCE TRAIN PATH <{lance_train_path}>')
                logger.info(f'NUM SPECTRA CONTAINED IN TRAIN DATASET {len(self.train_dataset)}')
                
            if self.valid_dataset_paths is not None:
                self.valid_dataset = self.create_new_lance( self.valid_dataset_paths,
                                                           lance_valid_path)
                
                logger.info(f'LANCE VALID PATH {lance_valid_path}')
                logger.info(f'NUM SPECTRA CONTAINED IN VAL DATASET {len(self.valid_dataset)}')
                
                
        if stage in (None, "test"):
            if self.test_dataset_paths is not None:
                try:
                    #logger.info('Trying annotated version of predict to include true sequence')
                    self.test_dataset = self.create_new_lance( self.test_dataset_paths,
                                                              lance_path=lance_test_path,
                                                              annotated=annotated, #True 
                                                              )
                except:
                    #logger.info('Annotated version failed... back to unannotated version')
                    self.test_dataset = self.create_new_lance( self.test_dataset_paths,
                                                              lance_path=lance_test_path,
                                                              annotated=False
                                                              )
                    
                logger.info(f'NUM SPECTRA CONTAINED IN TEST DATASET {len(self.test_dataset)}')
                #logger.info(f'PATHS CONTAINED IN TEST DATASET {self.test_dataset.peak_files}')


    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the training DataLoader."""
        logger.info(f'TRAIN DATALOADER n_workers {self.n_workers}')
        try:
            loader = self.train_dataset.loader(batch_size=self.train_batch_size, 
                                         num_workers=self.n_workers,
                                         precision=torch.float32, #torch.float16,
                                         pin_memory=True,
                                       #  persistent_workers=True,
                                        )
        except:
            loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the validation DataLoader."""
        logger.info(f'VAL DATALOADER n_workers {self.n_workers}')
        return self.valid_dataset.loader(batch_size=self.eval_batch_size,
                                         num_workers=self.n_workers,
                                         precision=torch.float32, #torch.float16,
                                         pin_memory=True,
                                        # persistent_workers=True,
                                        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the test DataLoader."""
        return self.test_dataset.loader(batch_size=self.eval_batch_size,
                                        num_workers=self.n_workers,
                                        precision=torch.float32, #torch.float16,
                                        pin_memory=True,
                                        #persistent_workers=True,
                                       )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the predict DataLoader."""
        return self.test_dataset.loader(batch_size=self.eval_batch_size, 
                                        num_workers=self.n_workers,
                                        precision=torch.float32, #torch.float16,
                                        pin_memory=False,
                                        #persistent_workers=True,
                                       )


