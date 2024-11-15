![genetic_prompt_opt_icon](genetic_prompt_opt_icon.png)

# Introduction
Workflow for embedding clinical trials in order to find semantically similar trials or for topic
finding

# Dependencies
The package has the following dependencies:
Currently the pipeline only works for openai/chatgpt and also uses the bulk request api
for the submission of text to embed

- numpy
- openai
- sklearn
- pydantic
- torch
- ijson
- faiss
- skopt

# Usage

```
pip install git+https://github.com/MatthewCorney/GeneticPromptOpt.git
```
## Define our base classes for the input and output
Definging the outptut format is important so that openai will try to return data in the correct format
