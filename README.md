# BOSHNAS

This repository contains the relevant training and testing scipts for the BOSHNAS tool for efficient neural architecture search. It is based on the [naszilla/naszilla](https://github.com/naszilla/naszilla) repo.

## Table of Contents

- [Environment setup](#environment-setup)
- [Run experiments](#run-experiments)
- [Developer](#developer)
- [Cite this work](#cite-this-work)
- [License](#license)

## Environment setup

Clone this repository and install its requirements (which includes [nasbench](https://github.com/google-research/nasbench), 
[nas-bench-201](https://github.com/D-X-Y/NAS-Bench-201), and [nasbench301](https://github.com/automl/nasbench301)). It may take a few minutes.

```shell
git clone https://github.com/JHA-Lab/boshnas.git
cd boshnas
```

For [pip](https://pip.pypa.io/en/stable/), use the `requirements.txt` file.

```shell
cat requirements.txt | xargs -n 1 -L 1 pip install
pip install -e .
```

If you use [conda](https://docs.conda.io/en/latest/), you can use the environment setup script.

```shell
source env_setup.sh
```

Next, download the NAS benchmark dataset (either with the terminal command below, or from the website ([nasbench](https://github.com/google-research/nasbench)):

```shell
wget https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
```

## Run experiments

To run various NAS algorithms on the NASBench-101 dataset, use the script `naszilla/run_experiments.py`.

```shell
cd naszilla
python run_experiments.py --algo_params <algo_params> --queries 50 --trials 1
```

Here, `<algo_params>` can be one of: `simple_algos`, `all_algos`, `sota_algos` (which only contains BANANAS and BOSHNAS), `local_search_variants`, `random`, `evolution`, `bananas`, `gp_bo`, `dngo`, `bohamiann`, `local_search`, `nasbot`, `gcn_predictor`, and `boshnas`. Other evaluation flags can be seen by running the command `python run_experiments.py --help`.

## Developers

[Shikhar Tuli](https://github.com/shikhartuli) and [Shreshth Tuli](https://github.com/shreshthtuli). For any questions, comments or suggestions, please reach out at [stuli@princeton.edu](mailto:stuli@princeton.edu).

## Cite this work

Cite our work using the following bitex entry:

```bibtex
@article{tuli2022jair,
      title={{FlexiBERT}: Are Current Transformer Architectures too Homogeneous and Rigid?}, 
      author={Tuli, Shikhar and Dedhia, Bhishma and Tuli, Shreshth and Jha, Niraj K.},
      year={2022},
      eprint={2205.11656},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

This work is used in [jha-lab/txf_design-space](https://github.com/jha-lab/txf_design-space). Other works that leverage BOSHNAS and its extensions include:

```bibtex
@article{tuli2022codebench,
  title={CODEBench: A Neural Architecture and Hardware Accelerator Co-Design Framework},
  author={Tuli, Shikhar and Li, Chia-Hao and Sharma, Ritvik and Jha, Niraj K},
  journal={ACM Transactions on Embedded Computing Systems},
  publisher={ACM New York, NY}
}
```

```bibtex
@article{tuli2023transcode,
      title={{TransCODE}: Co-design of Transformers and Accelerators for Efficient Training and Inference}, 
      author={Tuli, Shikhar and Jha, Niraj K.},
      year={2023},
      eprint={2303.14882},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

BSD-3-Clause. 
Copyright (c) 2022, Shikhar Tuli and Jha Lab.
All rights reserved.

See License file for more details.
