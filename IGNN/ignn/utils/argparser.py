"""Parse All Model Args"""

import argparse
import os
from pathlib import Path, PurePath
from typing import Dict

import yaml
from the_utils import tab_printer

type_map = {"int": int, "str": str, "float": float, "bool": bool}

#: Info of the models supported.
models: Dict = {
    "pca_kmeans": {
        "name": "PCA",
        "description": "PCA with Kmeans.",
        "paper url": "",
        "source code": "",
    },
}


def get_pkg_dir():
    return os.path.abspath(
        f"{os.path.dirname(os.path.realpath(__file__))}/..",
    )


def set_args_wrt_dataset(args, args_dict):
    """load experiment's changed model hyparamenter from yaml file

    Parameters
    ----------
    args : ArgumentParser
        ArgumentParser
    args_dict : dict
        dict of arguments from hyparamenter yaml file
    """
    # with open(args_dict["config"], "r", encoding="utf-8") as f:
    #     cfg_file = f.read()
    cfg_dict = read_configs(args_dict["model"])[args_dict["dataset"]]
    for k, v in cfg_dict.items():
        args.__setattr__(k, v)
    return args


def read_configs(model: str = None) -> Dict:
    pkg_dir = get_pkg_dir()
    config_path: PurePath = Path(f"{pkg_dir}/configs/{model}.yaml")
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        args = yaml.safe_load(f)
    return args


def set_subparser(model: str, _parser: argparse.ArgumentParser) -> None:
    args = read_configs(model)
    for key, val in args.items():
        keys = val.keys()
        default_val = val["default"] if "default" in keys else None
        type_val = type_map[val["type"]] if "type" in keys else type(val["default"])
        nargs_val = val["nargs"] if "nargs" in keys else None
        _parser.add_argument(
            f"--{key}",
            type=type_val,
            default=default_val,
            help=val["help"],
            nargs=nargs_val,
        )


def parse_all_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="HeteroGNN",
        description="Parameters for Heterophilous GNNs",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        help="Dataset used in the experiment",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="ID(s) of gpu used by cuda",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./data",
        help="Path to store the dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4096,
        help="Random seed. Defaults to 4096.",
    )

    subparsers = parser.add_subparsers(dest="model", help="sub-command help")

    for _model, items in models.items():
        # pylint: disable=invalid-sequence-index
        _help = items["description"]
        _parser = subparsers.add_parser(
            _model,
            help=f"Run Graph Clustering on {_help}",
        )
        set_subparser(_model, _parser)

    args = parser.parse_args()

    tab_printer(args)

    return args


def get_default_args(model: str) -> Dict:
    """Get default args of any model supported.

    Args:
        model (str): name of the model.

    Returns:
        Dict: the default args of the model.
    """
    _args = read_configs(model)
    args = {}
    for key, val in _args.items():
        keys = val.keys()
        default_val = val["default"] if "default" in keys else None
        args[key] = default_val

    return args
