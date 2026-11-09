This directory contains an unmodified, selected author release from
[tianhanz/DNN-Models-for-Chemical-Kinetics](https://github.com/tianhanz/DNN-Models-for-Chemical-Kinetics),
commit `bc07e5a210ba357b8a5350b460a751a68a002174`.

The upstream Apache-2.0 license is in `license`. Individual Git blob IDs and
SHA256 values are recorded in `source_manifest.json`. The mechanism and both
pretrained checkpoints are retained exactly as released. No upstream training
data were downloaded or substituted.

The release implements the 2022 multiscale-sampling ANN demonstration. Although
its README also cites Wang et al. (2025), the supplied `modelclass.py` does not
implement ANN-hard. Applying that paper's elemental projection is a separately
labelled diagnostic in our runner, not a claim to reproduce its full 9-species
model or its original training process.
