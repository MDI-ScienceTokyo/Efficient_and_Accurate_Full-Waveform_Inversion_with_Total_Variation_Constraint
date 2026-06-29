# Inner-Loop-Free Total-Variation-Constrained Full-Waveform Inversion

Shingo Takemoto and Shunsuke Ono

MDI Lab, Institute of Science Tokyo, Japan

## How to use
required [poetry](https://python-poetry.org/).

If you don't use poetry, install the libraries listed in `pyproject.toml` file.

E2E execution is as follows:
```bash
poetry install
poetry run inv download-salt-and-overthrust-models
poetry run python src/box-TV-constrained-FWI.py
```

The execution environment is as follows:
- CPU: 13th Gen Intel(R) Core(TM) i9-13900K
- Memory: 32GB

If computing resources are limited, consider reducing `num_parallels` in `src/box-TV-constrained-FWI.py`.

## Links 
- [official](https://www.mdi.c.titech.ac.jp/publications/fwiwtv)
- [preprint (arXiv)](https://arxiv.org/abs/2501.08210)
- [source code(GitHub)](https://github.com/MDI-TokyoTech/Convergent_Primal-Dual_Plug-and-Play_Image_Restoration_A_General_Algorithm_and_Applications)

## Abstract
This paper proposes a computationally efficient algorithm to address the Full-Waveform Inversion (FWI) problem with a Total Variation (TV) constraint, designed to accurately reconstruct subsurface properties from seismic signal. FWI, as an ill-posed inverse problem, requires effective regularizations or constraints to ensure accurate and stable solutions. Among these, the TV constraint is widely known as a powerful prior for modeling the piecewise smooth structure of subsurface properties. However, solving the optimization problem is challenging because of the nonlinear observation process combined with the non-smoothness of the TV constraint. Conventional methods rely on inner loops and/or approximations, which lead to high computational cost and/or inappropriate solutions. To address these limitations, we develop a novel algorithm based on a primal-dual splitting method, achieving computational efficiency by eliminating inner loops and ensuring high accuracy by avoiding approximations. We also demonstrate the effectiveness of the proposed method through experiments using the SEG/EAGE Salt Models. The source code will be available at https://www.mdi.comp.isct.ac.jp/publications/fwiwtv. 

## Experimental Results
![Image](https://github.com/user-attachments/assets/4713a589-f4a3-40fb-9ffd-8c679179ff0c)

