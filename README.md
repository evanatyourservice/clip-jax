# CLIP-JAX

This repository is used to train vision models with JAX:

- many types of model architectures
- any sharding strategy
- training with constrastive loss such as [CLIP](https://arxiv.org/abs/2103.00020), [chunked sigmoid loss](https://arxiv.org/abs/2303.15343) or captioning loss such as [CapPa](https://arxiv.org/abs/2306.07915)
- downstream fine-tuning

Refer to the report "[CapPa: Training vision models as captioners](https://wandb.ai/craiyon/cappa-jax/reports/CapPa-Training-vision-models-as-captioners--Vmlldzo4NDUyNDUz)" for the open-source reproduction of CapPa.

## Installation

```bash
pip install clip-jax
```

Note: this package is currently under active development, install from source for latest version.

For example:

```bash
git clone https://github.com/evanatyourservice/clip-jax.git && \
cd clip-jax && \
pip install -U pip && \
pip install -e '.[dev]' && \
pip install --force-reinstall --upgrade --no-cache-dir 'jax[tpu]==0.4.34' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html flax==0.10.0 && \
pip install 'numpy<2'
```

## Usage

### Use a trained model

Refer to [`utils/demo_cappa.ipynb`](utils/demo_cappa.ipynb).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/borisdayma/clip-jax/blob/main/utils/demo_cappa.ipynb)

You can find the [model weights](https://huggingface.co/boris/cappa-large-patch16-256-jax) on Hugging Face.

### Download training data

You can download training data from [DataComp](https://github.com/mlfoundations/datacomp):

```bash
# clone and install datacomp

# download data
python download_upstream.py \
    --scale small --data_dir gs://my_bucket/datacomp/small metadata_dir metadata \
    --image_size 256 --resize_mode center_crop --skip_bbox_blurring --no_resize_only_if_bigger \
    --encode_format webp --output_format tfrecord
```

Alternatively, you can use your own dataset. In that case you should use [img2dataset](https://github.com/rom1504/img2dataset) with `output_format="tfrecord"`.

### Train a model

Use [`training/train.py`](training/train.py) to train a model:

Here is an example command to train a model on a TPU v3-8:

```bash
python train.py \
    --assert_TPU_available \
    --output_dir /home/evanatyourservice/clip-jax/training/trained_model --overwrite_output_dir --checkpoints_to_keep 1 \
    --config_name ../configs/mini-patch16-cappa.json \
    --tokenizer_name boris/cappa-large-patch16-256-jax \
    --unroll 100 \
    --train_folder ./datacomp1b_train.pkl --valid_folder ./datacomp1b_valid.pkl \
    --image_crop_resize 256 \
    --key_caption caption_normalized \
    --do_train --do_eval \
    --n_predict 128 --n_predict_batch 8 \
    --dtype bfloat16 --float32_logits \
    --remat_policy none \
    --learning_rate 1.0e-4 --warmup_steps 2000 --lr_offset 0 \
    --batch_size_per_node 512 --gradient_accumulation 1 --num_train_epochs 2 --vision_projection_only False \
    --valid_batch_size_per_node 256 --weight_decay 0.0 \
    --optim distributed_shampoo --beta1 0.9 --beta2 0.99 --preconditioning_compute_steps 20 --block_size_text 1024 --block_size_vision 1024 --nesterov --graft_type rmsprop_normalized \
    --mp_devices 1 --shard_shampoo_across 2d --activation_partitioning_dims 1 --parameter_partitioning_dims 1 \
    --logging_steps 100 --eval_steps 2000 --save_steps 2000
```

## Acknowledgements

- [Lucas Beyer](https://twitter.com/giffmana) for helping with clarifications on the [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) paper and [Image Captioners Are Scalable Vision Learners Too](https://arxiv.org/abs/2306.07915)
- [Timothée Darcet](https://twitter.com/TimDarcet) for helping with clarifications on the [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) paper
- 🤗 Hugging Face for reference implementation of CLIP
- Google [TPU Research Cloud (TRC) program](https://sites.research.google/trc/) for providing computing resources
- [Weights & Biases](https://wandb.com/) for providing the infrastructure for experiment tracking and model management
- [Big Vision Github Repository](https://github.com/google-research/big_vision) for reference code of many papers

## Citations

```bibtex
@misc{radford2021learning,
      title={Learning Transferable Visual Models From Natural Language Supervision},
      author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
      year={2021},
      eprint={2103.00020},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@misc{zhai2023sigmoid,
      title={Sigmoid Loss for Language Image Pre-Training},
      author={Xiaohua Zhai and Basil Mustafa and Alexander Kolesnikov and Lucas Beyer},
      year={2023},
      eprint={2303.15343},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@misc{zhai2022scaling,
      title={Scaling Vision Transformers}, 
      author={Xiaohua Zhai and Alexander Kolesnikov and Neil Houlsby and Lucas Beyer},
      year={2022},
      eprint={2106.04560},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@misc{tschannen2023image,
      title={Image Captioners Are Scalable Vision Learners Too}, 
      author={Michael Tschannen and Manoj Kumar and Andreas Steiner and Xiaohua Zhai and Neil Houlsby and Lucas Beyer},
      year={2023},
      eprint={2306.07915},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@misc{darcet2023vision,
      title={Vision Transformers Need Registers}, 
      author={Timothée Darcet and Maxime Oquab and Julien Mairal and Piotr Bojanowski},
      year={2023},
      eprint={2309.16588},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@misc{dehghani2023patch,
      title={Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution}, 
      author={Mostafa Dehghani and Basil Mustafa and Josip Djolonga and Jonathan Heek and Matthias Minderer and Mathilde Caron and Andreas Steiner and Joan Puigcerver and Robert Geirhos and Ibrahim Alabdulmohsin and Avital Oliver and Piotr Padlewski and Alexey Gritsenko and Mario Lučić and Neil Houlsby},
      year={2023},
      eprint={2307.06304},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@misc{mckinzie2024mm1,
      title={MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training}, 
      author={Brandon McKinzie and Zhe Gan and Jean-Philippe Fauconnier and Sam Dodge and Bowen Zhang and Philipp Dufter and Dhruti Shah and Xianzhi Du and Futang Peng and Floris Weers and Anton Belyi and Haotian Zhang and Karanjeet Singh and Doug Kang and Ankur Jain and Hongyu Hè and Max Schwarzer and Tom Gunter and Xiang Kong and Aonan Zhang and Jianyu Wang and Chong Wang and Nan Du and Tao Lei and Sam Wiseman and Guoli Yin and Mark Lee and Zirui Wang and Ruoming Pang and Peter Grasch and Alexander Toshev and Yinfei Yang},
      year={2024},
      eprint={2403.09611},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@misc{hsieh2023sugarcrepefixinghackablebenchmarks,
      title={SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality}, 
      author={Cheng-Yu Hsieh and Jieyu Zhang and Zixian Ma and Aniruddha Kembhavi and Ranjay Krishna},
      year={2023},
      eprint={2306.14610},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2306.14610}, 
}
```
