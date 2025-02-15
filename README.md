# Exploring the Potential of Encoder-free Architectures in 3D LMMs

Official repository for the paper "[Exploring the Potential of Encoder-free Architectures in 3D LMMs](https://arxiv.org/pdf/2502.09620v1)".

[[📖 Paper](https://arxiv.org/pdf/2502.09620v1)] [[🤗 HF Checkpoints for stage1](https://huggingface.co/IvanTang/ENEL/tree/main)] 

## 🏠 About
<!-- ![Teaser](assets/teaser.png) -->
<div style="text-align: center;">
    <img src="assets/teaser.png" alt="Solution_Teaser" width=100% >
</div>
We introduce <b>ENEL, an Encoder-free 3D Large Language Model capable of overcoming the challenges posed by encoder-based architectures</b>, including the inability to <b>adapt to varying point cloud resolutions</b> and the failure of encoder-extracted point features to <b>meet the semantic needs of Large Language Models</b>. Building upon PointLLM, we conduct a comprehensive investigation into how <b>the LLM can assume the role of the 3D encoder</b>. Based on the PointLLM dataset, our 7B model is evaluated across <b>three benchmark tasks: generative 3D object classification, 3D object captioning, and 3D VQA, with assessments performed using GPT-4 scoring and traditional metrics.</b>

## 🔥 News
- [2023-02-13] We release the codes for training in the pre-training stage with corresponding checkpoints and the codes for evaluation.
- [2025-02-13] We release the [paper](https://arxiv.org/pdf/2502.09620v1) of ENEL;

<!-- contents with emoji -->
## 📋 Contents
- [💬 Dialogue Examples](#-dialogue-examples)
- [🔍 Overview](#-overview)
- [📦 Training and Evaluation](#-training-and-evaluation)
- [📝 TODO List](#-todo-list)
- [🔗 Citation](#-citation)
- [📄 License](#-license)
- [📚 Related Work](#-related-work)


## 💬 Dialogue Examples
| Dialogue 1 
| :-: | 
| <img width="100%" src="assets/output.png"> |


## 🔍 Overview

### Model
<p align="center">
  <img src="assets/Pipeline.png" align="center" width="100%">
</p>
The encoder-free 3D LMM directly utilizes a token embedding module to convert point cloud data into discrete point tokens, which are then concatenated with text tokens to serve as input to the LLM. To assume the role of the encoder, the LLM is guided to extract high-level semantic features of the point clouds and acquire multi-level knowledge from both global and local perspectives. 

### Experiment Results
Please refer to our paper for more results.
<p align="center">
  <img src="assets/result.png" align="center" width="100%">
</p>

### Reminder of the Model Zoo
In [https://huggingface.co/IvanTang/ENEL/tree/main](https://huggingface.co/IvanTang/ENEL/tree/main), to adapt to different paths, please modify the attributes: _name_or_path in the config.json file and special_tokens_map_file in the tokenizer_config.json file.


## 📦 Training and Evaluation
### Installation
To start: 
1. Clone this repository.
```bash
https://github.com/Ivan-Tang-3D/ENEL.git
cd ENEL
```
2. Install packages
```bash
conda create -n ENEL python=3.10 -y
conda activate ENEL
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# * for training
pip install ninja
pip install flash-attn

# * for chamfer_dist
git clone https://github.com/Pang-Yatian/Point-MAE.git
cd ./extensions/chamfer_dist
python setup.py install --user
```

### Data Preparation
#### Objaverse Training Data
1. Download the two compressed files of 660K Objaverse colored point clouds [here](https://huggingface.co/datasets/RunsenXu/PointLLM/tree/main). They require about 77GB of storage space.
2. Run the following command to merge the two files into one and uncompress it. This will produce a folder named `8192_npy` containing 660K point cloud files named `{Objaverse_ID}_8192.npy`. Each file is a numpy array with dimensions (8192, 6), where the first three dimensions are `xyz` and the last three dimensions are `rgb` in [0, 1] range.
```bash
cat Objaverse_660K_8192_npy_split_a* > Objaverse_660K_8192_npy.tar.gz
tar -xvf Objaverse_660K_8192_npy.tar.gz
```
3. In `ENEL` folder, create a folder `data` and create a soft link to the uncompressed file in the directory.
```bash
cd ENEL
mkdir data
ln -s /path/to/8192_npy data/objaverse_data
```

#### Instruction-Following Data
1. In `ENEL/data` folder, create a directory named `anno_data`.
2. Our instruction-following data, including both the simple-description and complex instructions, can be downloaded [here](https://huggingface.co/datasets/RunsenXu/PointLLM). If you have difficulty downloading the data (e.g. network issue), please email the authors.
- The simple-description data has 660K samples and the complex instructions have 70K samples.
- Both training data are based on the Objaverse dataset.
- The complex instructions are generated with GPT-4.
3. Put the data files in the `anno_data` directory. The directory should look like this:
```bash
ENEL/data/anno_data
├── PointLLM_brief_description_660K_filtered.json
├── PointLLM_brief_description_660K.json
└── PointLLM_complex_instruction_70K.json
```
4. Note, the `PointLLM_brief_description_660K_filtered.json` is filtered from `PointLLM_brief_description_660K.json` by removing the 3000 objects we reserved as the validation set. 

#### Evaluation Data
1. Download the referencing GT `PointLLM_brief_description_val_200_GT.json` we use for the benchmarks on Objaverse dataset [here](https://huggingface.co/datasets/RunsenXu/PointLLM/blob/main/PointLLM_brief_description_val_200_GT.json), and put it in `ENEL/data/anno_data`.

### Training
#### Download the Initial LLM Weight
1. In `ENEL` folder, create a directory named `checkpoints`.
2. Download the pre-trained LLM: [
PointLLM_7B_v1.1_init](https://huggingface.co/RunsenXu/PointLLM_7B_v1.1_init/tree/main). Put them in the `checkpoints` directory.

#### Start Training
1. For stage-1 training, simply run:
```bash
cd ENEL
scripts/ENEL_train_stage1.sh
```

### Evaluation
#### Inferencing
1. Run the following commands to infer the results.
2. Different commands for inferencing on different benchmarks:
```bash
MODEL_NAME=
LOG_SUFFIX=
LOG_DIR="/ENEL/new_eval_logs"
LOG_EDIR="/ENEL/new_eval_logs"

export PYTHONPATH="/ENEL:$PYTHONPATH"

# Object captioning on Objaverse
CUDA_VISIBLE_DEVICES=1 python pointllm/eval/eval_objaverse.py --model_name $MODEL_NAME --task_type captioning --prompt_index 2 > $LOG_EDIR/try_obj_${LOG_SUFFIX}.log 2>&1 &

# Open Vocabulary Classification on Objaverse
CUDA_VISIBLE_DEVICES=2 python pointllm/eval/eval_objaverse.py  --model_name $MODEL_NAME --task_type classification --prompt_index 0 > $LOG_EDIR/try_objcls_${LOG_SUFFIX}.log 2>&1 &
```
3. Please check the default command-line arguments of these two scripts. You can specify different prompts, data paths, and other parameters. 
4. After inferencing, the results will be saved in `{model_name}/evaluation` as a dict with the following format:
```bash
{
  "prompt": "",
  "results": [
    {
      "object_id": "",
      "ground_truth": "", 
      "model_output": "",
      "label_name": "" # only for classification on modelnet40
    }
  ]
}
```

#### ChatGPT/GPT-4 Evaluation
1. Get your OpenAI API key at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys).
2. Please set the OpenAI API Key in the 40th line of [https://github.com/Ivan-Tang-3D/ENEL/blob/main/pointllm/eval/utils.py](https://github.com/Ivan-Tang-3D/ENEL/blob/main/pointllm/eval/utils.py).
3. Run the following commands to evaluate the model outputs in parallel with ChatGPT/GPT-4 (which cost approximately $1.5 to $2.2 USD).
```bash
export PYTHONPATH="/ENEL:$PYTHONPATH"

# Open Vocabulary Classification on Objaverse
python pointllm/eval/evaluator.py --results_path /path/to/model_output --model_type gpt-4-0613 --eval_type open-free-form-classification --parallel --num_workers 15

# Object captioning on Objaverse
python pointllm/eval/evaluator.py --results_path /path/to/model_output --model_type gpt-4-0613 --eval_type object-captioning --parallel --num_workers 15
```
3. The evaluation script supports interruption and resumption. You can interrupt the evaluation process at any time by using `Ctrl+C`. This will save the temporary results. If an error occurs during the evaluation, the script will also save the current state. You can resume the evaluation from where it left off by running the same command again.
4. The evaluation results will be saved in `{model_name}/evaluation` as another dict.
Some of the metrics are explained as follows:
```bash
"average_score": The GPT-evaluated captioning score we report in our paper.
"accuracy": The classification accuracy we report in our paper, including random choices made by ChatGPT when model outputs are vague or ambiguous and ChatGPT outputs "INVALID".
"clean_accuracy": The classification accuracy after removing those "INVALID" outputs.
"total_predictions": The number of predictions.
"correct_predictions": The number of correct predictions.
"invalid_responses": The number of "INVALID" outputs by ChatGPT.

# Some other statistics for calling OpenAI API
"prompt_tokens": The total number of tokens of the prompts for ChatGPT/GPT-4.
"completion_tokens": The total number of tokens of the completion results from ChatGPT/GPT-4.
"GPT_cost": The API cost of the whole evaluation process, in US Dollars 💵.
```

#### Traditional Metric Evaluation
1. For the object captioning task, run the following command to evaluate model outputs with traditional metrics including BLEU, ROUGE, METEOR, Sentence-BERT, and SimCSE.
```bash
export PYTHONPATH="/ENEL:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python pointllm/eval/traditional_evaluator.py --results_path /path/to/model_captioning_output
```


## 📝 TODO List
- [x] Add training codes for stage1 with checkpoints.
- [x] Add evaluation&inferencing codes.
- [ ] Add training codes for stage2.

## 🔗 Citation

If you find our work and this codebase helpful, please consider starring this repo 🌟 and cite:
```bibtex
@misc{tang2025exploringpotentialencoderfreearchitectures,
      title={Exploring the Potential of Encoder-free Architectures in 3D LMMs}, 
      author={Yiwen Tang and Zoey Guo and Zhuhao Wang and Ray Zhang and Qizhi Chen and Junli Liu and Delin Qu and Zhigang Wang and Dong Wang and Xuelong Li and Bin Zhao},
      year={2025},
      eprint={2502.09620},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.09620}, 
}
```


## 📄 License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## 👏 Acknowledgements
- [Point-Bind & Point-LLM](https://arxiv.org/abs/2309.00615)
- [PointLLM](https://arxiv.org/abs/2308.16911)
- [ShapeLLM](https://arxiv.org/abs/2402.17766)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Vicuna](https://github.com/lm-sys/FastChat)
- [Objaverse](https://objaverse.allenai.org)
- [Cap3D](https://github.com/crockwell/Cap3D/)
