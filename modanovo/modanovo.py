"""The command line entry point."""

import datetime
import functools
import logging
import os
import re
import shutil
import glob
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple, Iterable, List

warnings.formatwarning = lambda message, category, *args, **kwargs: (
    f"{category.__name__}: {message}"
)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    ".*Consider increasing the value of the `num_workers` argument*",
)
warnings.filterwarnings(
    "ignore",
    ".*The PyTorch API of nested tensors is in prototype stage*",
)
warnings.filterwarnings(
    "ignore",
    ".*Converting mask without torch.bool dtype to bool*",
)

import appdirs
import depthcharge
import github
import lightning
import requests
import rich_click as click
import torch
import tqdm
from lightning.pytorch import seed_everything

from . import __version__
from . import utils
from .denovo import ModelRunner
from .config import Config

logger = logging.getLogger("modanovo")
click.rich_click.USE_MARKDOWN = True
click.rich_click.STYLE_HELPTEXT = ""
click.rich_click.SHOW_ARGUMENTS = True


class _SharedParams(click.RichCommand):
    """Options shared between most commands"""

    def __init__(self, *args, **kwargs) -> None:
        """Define shared options."""
        super().__init__(*args, **kwargs)
        self.params += [
            click.Option(
                ("-m", "--model"),
                help="""
                The model weights (.ckpt file).
                """,
                type=click.Path(exists=True, dir_okay=False),
            ),
            click.Option(
                ("-o", "--output"),
                help="The mzTab file to which results will be written.",
                type=click.Path(dir_okay=False),
            ),
            click.Option(
                ("-c", "--config"),
                help="""
                The YAML configuration file overriding the default options.
                """,
                type=click.Path(exists=True, dir_okay=False),
            ),
            click.Option(
                ("-v", "--verbosity"),
                help="""
                Set the verbosity of console logging messages. Log files are
                always set to 'debug'.
                """,
                type=click.Choice(
                    ["debug", "info", "warning", "error"],
                    case_sensitive=False,
                ),
                default="info",
            ),
        ]


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main() -> None:
    """######

    """
    return


@main.command(cls=_SharedParams)
@click.argument(
    "peak_path",
    required=True,
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False),
)
def sequence(
    peak_path: Tuple[str],
    model: Optional[str],
    config: Optional[str],
    output: Optional[str],
    verbosity: str,
) -> None:
    """De novo sequence peptides from tandem mass spectra.

    PEAK_PATH must be one or more mzMl, mzXML, or MGF files from which
    to sequence peptides.
    """
    output = setup_logging(output, verbosity)
    config, model = setup_model(model, config, output, False)
    with ModelRunner(config, model) as runner:
        logger.info("Sequencing peptides from:")
        for peak_file in peak_path:
            logger.info("  %s", peak_file)

        runner.predict(peak_path, output)

    logger.info("DONE!")


@main.command(cls=_SharedParams)
@click.argument(
    "annotated_peak_path",
    required=True,
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False),
)
def evaluate(
    annotated_peak_path: Tuple[str],
    model: Optional[str],
    config: Optional[str],
    output: Optional[str],
    verbosity: str,
) -> None:
    """Evaluate de novo peptide sequencing performance.

    ANNOTATED_PEAK_PATH must be one or more annoated MGF files,
    such as those provided by MassIVE-KB.
    """
    output = setup_logging(output, verbosity)
    config, model = setup_model(model, config, output, False)
    with ModelRunner(config, model) as runner:
        logger.info("Sequencing and evaluating peptides from:")
        for peak_file in annotated_peak_path:
            logger.info("  %s", peak_file)

        runner.evaluate(annotated_peak_path)

    logger.info("DONE!")


@main.command(cls=_SharedParams)
@click.argument(
    "train_peak_path",
    required=True,
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "-p",
    "--validation_peak_path",
    help="""
    An annotated MGF file for validation, like from MassIVE-KB. Use this
    option multiple times to specify multiple files.
    """,
    required=True,
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
)
def train(
    train_peak_path: Tuple[str],
    validation_peak_path: str,#Tuple[str],
    model: Optional[str],
    config: Optional[str],
    output: Optional[str],
    verbosity: str,
) -> None:
    """Train a model on your own data.

    TRAIN_PEAK_PATH must be one or more annoated MGF files, such as those
    provided by MassIVE-KB, from which to train a new Casnovo model.
    """
    output = setup_logging(output, verbosity)
    config, model = setup_model(model, config, output, True)
    
    print('train_peak_path', type(train_peak_path), train_peak_path)
    print('validation_peak_path', type(validation_peak_path), validation_peak_path)
    #train_peak_path = _get_peak_filenames(train_peak_path)
    #validation_peak_path = _get_peak_filenames(validation_peak_path)
    
    with ModelRunner(config, model) as runner:
        logger.info("Training a model from:")
        for peak_file in train_peak_path:
            logger.info("  %s", peak_file)

        logger.info("Using the following validation files:")
        for peak_file in validation_peak_path:
            logger.info("  %s", peak_file)

        runner.train(train_peak_path, validation_peak_path)

    logger.info("DONE!")


@main.command()
def version() -> None:
    """Get the version information"""
    versions = [
        f": {__version__}",
        f"Depthcharge: {depthcharge.__version__}",
        f"Lightning: {lightning.__version__}",
        f"PyTorch: {torch.__version__}",
    ]
    sys.stdout.write("\n".join(versions) + "\n")


@main.command()
@click.option(
    "-o",
    "--output",
    help="The output configuration file.",
    default="config.yaml",
    type=click.Path(dir_okay=False),
)
def configure(output: str) -> None:
    """Generate a configuration file to customize.

    The configuration file is in the YAML format.
    """
    Config.copy_default(output)
    output = setup_logging(output, "info")
    logger.info(f"Wrote {output}\n")


def setup_logging(
    output: Optional[str],
    verbosity: str,
) -> Path:
    """Set up the logger.

    Logging occurs to the command-line and to the given log file.

    Parameters
    ----------
    output : Optional[str]
        The provided output file name.
    verbosity : str
        The logging level to use in the console.

    Return
    ------
    output : Path
        The output file path.
    """
    if output is None:
        os.makedirs('modanovo_logs', exist_ok=True)
        output = f"modanovo_logs/logs_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    output = Path(output).expanduser().resolve()

    logging_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    # Configure logging.
    logging.captureWarnings(True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    warnings_logger = logging.getLogger("py.warnings")

    # Formatters for file vs console:
    console_formatter = logging.Formatter("{levelname}: {message}", style="{")
    log_formatter = logging.Formatter(
        "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : "
        "{message}",
        style="{",
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging_levels[verbosity.lower()])
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    warnings_logger.addHandler(console_handler)
    file_handler = logging.FileHandler(output.with_suffix(".log"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    warnings_logger.addHandler(file_handler)

    # Disable dependency non-critical log messages.
    logging.getLogger("depthcharge").setLevel(
        logging_levels[verbosity.lower()]
    )
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("github").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return output


def setup_model(
    model: Optional[str],
    config: Optional[str],
    output: Optional[Path],
    is_train: bool,
) -> Config:
    """Setup for most commands.

    Parameters
    ----------
    model : Optional[str]
        The provided model weights file.
    config : Optional[str]
        The provided configuration file.
    output : Optional[Path]
        The provided output file name.
    is_train : bool
        Are we training? If not, we need to retrieve weights when the model is
        None.

    Return
    ------
    config : Config
        The parsed configuration
    """
    # Read parameters from the config file.
    config = Config(config)
    seed_everything(seed=config["random_seed"], workers=True)


    # Log the active configuration.
    logger.info("version %s", str(__version__))
    logger.debug("model = %s", model)
    logger.debug("config = %s", config.file)
    logger.debug("output = %s", output)
    for key, value in config.items():
        logger.debug("%s = %s", str(key), str(value))

    return config, model

def _get_peak_filenames( paths: Iterable[str], 
                        supported_ext: Iterable[str] = (".mgf",)
) -> List[str]:
    """
    Get all matching peak file names from the path pattern.

    Performs cross-platform path expansion akin to the Unix shell (glob, expand
    user, expand vars).

    Parameters
    ----------
    paths : Iterable[str]
        The path pattern(s).
    supported_ext : Iterable[str]
        Extensions of supported peak file formats.

    Returns
    -------
    List[str]
        The peak file names matching the path pattern.
    """
    found_files = set()
    for path in paths:
        path = os.path.expanduser(path)
        path = os.path.expandvars(path)
        for fname in glob.glob(path, recursive=True):
            if Path(fname).suffix.lower() in supported_ext:
                found_files.add(fname)

    return sorted(list(found_files))

    
if __name__ == "__main__":
    main()
