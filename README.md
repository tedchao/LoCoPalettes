# LoCoPalettes

*LoCoPalettes*: Local Control for Palette-based Image Editing ([project page](https://cragl.cs.gmu.edu/locopalettes/))

*Computer Graphics Forum (CGF)*. Presented at *EGSR 2023*.

[*By* [Cheng-Kang Ted Chao](https://mason.gmu.edu/~cchao8/), [Jason Klein](https://www.linkedin.com/in/jason-adam-klein), [Jianchao Tan](https://scholar.google.com/citations?user=1Gywy80AAAAJ&hl=en), [Jose Echevarria](http://www.jiechevarria.com/), [Yotam Gingold](https://cragl.cs.gmu.edu/)] 

See [demo video](https://cragl.cs.gmu.edu/locopalettes/) for our editing framework.

## About

This repo is official code release for *LoCoPalettes*. 

The contribution of this work:
1. An approach to compute sparser weights, comprimising spatial coherence with sparsity, to achieve sparse color edits without much color leakage compared to state-of-the-art.
2. An optimization framework that operates on our proposed palette hierarchy, enabling semantic color editing via placing color constraints.

## Installation

You can install dependencies using either `conda` or `pip`.

### Conda

Install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
(Miniconda is faster to install.) Choose the 64-bit Python 3.x version. Launch the Anaconda shell from the Start menu and navigate to this directory.
Then:

    conda env create -f environment.yml
    conda activate sparse_edit

To update an already created environment if the `environment.yml` file changes, first activate and then run `conda env update --file environment.yml --prune`.
