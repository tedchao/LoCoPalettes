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

You can install dependencies using `conda`.

### Conda

Install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
(Miniconda is faster to install.) Choose the 64-bit Python 3.x version. Launch the Anaconda shell from the Start menu and navigate to this directory.
Then:

    conda env create -f environment.yml
    conda activate locopalettes

To update an already created environment if the `environment.yml` file changes, first activate and then run `conda env update --file environment.yml --prune`.

### Usage

First, we need to have features (.mat) files. To extract per-pixel features, please refer to [Aksoy's code](https://github.com/tedchao/SIGGRAPH18SSS) and run

    sh run_extract_feat.sh

Then, we need to extract panoptic segments (followed by guided filtering), run

    python panoptic.py <your_image>

This would create a folder named `<sss_your_image>` that contains segments under root -> classes -> instances hierarchy.

Then, run the below to extract all informations (i.e. palettes, weights, masks, activations, trees):

    python func/seg.py <your_image> <your_feature> <sss_your_image> --o s --m fea

The above code generates all informations needed alongside `GUI.py`,

Finally, run

    python GUI.py

After the GUI is loaded, follow the below order clicking:

(1) Load image
(2) Load features
(3) Load tree (choose arbitrary image from your `<sss_your_image>` folder)

## License

[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
