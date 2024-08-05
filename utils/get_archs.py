import os
import sys

sys.path.append(os.path.abspath("."))
from utils.import_file import import_file


def get_backbone_arch(arch: str):

    base_dir = os.path.join(os.getcwd(), "backbone_archs")
    all_modules = import_file(
        "",
        path=os.path.join(base_dir, f"{arch}.py"),
    )
    return all_modules.Backbone


def get_sampler_arch(arch: str):

    base_dir = os.path.join(os.getcwd(), "sampler_archs")
    all_modules = import_file(
        "",
        path=os.path.join(base_dir, f"{arch}.py"),
    )
    return all_modules.Sampler


def get_diffuser_arch(arch: str):

    base_dir = os.path.join(os.getcwd(), "diffuser_archs")
    all_modules = import_file(
        "",
        path=os.path.join(base_dir, f"{arch}.py"),
    )
    return all_modules.Diffuser
