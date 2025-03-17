# instruct tuning

This repo contains code for instruct tuning of LLMs.

I plan to use this repo to test out different instruct tuning techniques.


```bash
huggingface-cli login      
```

Llama 3.3 70B is used as the base model.

Determine your rig using this article: [Self-Hosting Llama 3.1 70B or Any 70B LLM Affordably](https://abhinand05.medium.com/self-hosting-llama-3-1-70b-or-any-70b-llm-affordably-2bd323d72f8d)


```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

```


LIMA is used as the instruction tuning dataset.

```python
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("GAIR/lima")

```

## Setting Up the Environment

To set up the Conda environment for this project, follow these steps:

1. **Create the Conda Environment**:
   ```bash
   conda create --name instruct-tuning-env python=3.10
   conda activate instruct-tuning-env
   ```

2. **Install Required Libraries**:
   ```bash
   conda install jupyter
   pip install transformers
   pip install datasets
   pip install peft
   pip install torch
   pip install accelerate
   pip install bitsandbytes
   pip install xformers
   pip install sentencepiece
   pip install langchain
   ```

3. **Launching Jupyter Notebook**:
   After activating the environment, you can start Jupyter Notebook by running:
   ```bash
   jupyter notebook
   ```

## Updating the Environment

To update the environment with new libraries, you can use the following command:
```bash
conda install <library-name>
```
Replace `<library-name>` with the name of the library you want to install.

## Running the Jupyter Notebook

To run the Jupyter Notebook, navigate to the directory containing the notebook and run:

## Updating the Conda Environment File

To update the Conda environment file (`environment.yml`) with new libraries or changes, follow these steps:

1. **Activate the Conda Environment**:
   Make sure you have activated the environment you want to update:
   ```bash
   conda activate instruct-tuning-env
   ```

2. **Install the New Library**:
   Install the new library or make changes to the environment:
   ```bash
   conda install <library-name>
   ```
   Replace `<library-name>` with the name of the library you want to install.

3. **Export the Updated Environment**:
   Export the updated environment to the `environment.yml` file:
   ```bash
   conda env export > environment.yml
   ```

This will update the `environment.yml` file with the current state of your Conda environment, including any new libraries or changes you have made.

## Restoring the Environment from `environment.yml`

To restore the Conda environment from the `environment.yml` file, follow these steps:

1. **Navigate to the Directory**:
   Navigate to the directory containing the `environment.yml` file:
   ```bash
   cd /path/to/directory
   ```

2. **Create the Environment**:
   Create the Conda environment from the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the Environment**:
   After the environment is created, activate it:
   ```bash
   conda activate instruct-tuning-env
   ```

This will restore the Conda environment with all the libraries and dependencies specified in the `environment.yml` file.
