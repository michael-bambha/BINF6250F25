#!/usr/bin/env python3

"""
File: Project01.py
Author: Michael Bambha
Date: 09-10-2025
Description: Project 1 -- VCF Parsing Implementation
"""
# pylint: disable=invalid-name

from typing import List
from collections import Counter
from pprint import pprint


def parse_line(vcf_string: str) -> List[str]:
    """Parse a line of a VCF file for the allele
    frequency AF_EXAC and return a list of diseases
    for that entry if the allele frequency is
    < 0.0001.

    Args:
        vcf_string (str): A non-metadata line of a VCF
        file (tab-delimited). Example:

        1	1014451	475281	C	T	.	.	AF_EXAC=0.0077226;CLNDN=Immunodeficiency_38_with_basal_ganglia_calcification

    Returns:
        List[str]: List of diseases found
    """

    attrs = {}

    try:
        info = vcf_string.split("\t")[7]
    except (
        IndexError
    ):  # in case the line is incorrectly formatted, we return an empty list
        return []

    for part in filter(
        None, info.strip().split(";")
    ):  # if .strip().split() returns "", filter will remove from list
        if "=" in part:
            k, v = part.split("=", 1)
            attrs[k.strip()] = v.strip()

    allele_freq = attrs.get("AF_EXAC")

    if allele_freq is None:
        return []

    if float(allele_freq) < 0.0001:
        return _filter_CLNDN(attrs.get("CLNDN"))

    return []


def _filter_CLNDN(clndn_str: str) -> List[str]:
    """Take in the value of CLNDN parsed out of a VCF file
    and return a list of diseases.

    Args:
        clndn_str (str): The value of the CLNDN parsed from
        key/value hierarchy of VCF formatted files. Contains
        a string of diseases separated by "|".
        Example: Myasthenic_syndrome|Breast_cancer

    Returns:
        List[str]: List of diseases parsed by the "|".
        Diseases that are "not_specified" or "not_provided" are
        not returned.
    """
    strings_to_filter = ["not_specified", "not_provided"]
    clndn_str_filtered = clndn_str.split("|")
    return [
        disease for disease in clndn_str_filtered if disease not in strings_to_filter
    ]


def read_file(file_name: str) -> Counter[str]:
    """Read in a VCF file and return a dictionary of
    diseases counts found to be associated with allele
    frequencies of < 0.0001. Counts represent the number
    that each associated disease was found across
    all variants.

    Args:
        file_name (str): VCF file to read

    Returns:
        Counter[str]: Counter object of disease counts
        from all variants. Example:
        {"Breast_cancer": 20, ...}
    """

    disease_count = Counter()
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            diseases = parse_line(line)
            if diseases:
                disease_count.update(diseases)

    return disease_count


if __name__ == "__main__":
    pprint(read_file("clinvar_20190923_short.vcf"))
