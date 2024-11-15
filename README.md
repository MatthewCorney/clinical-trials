![genetic_prompt_opt_icon](genetic_prompt_opt_icon.png)

# Introduction
Workflow for embedding clinical trials in order to find semantically similar trials or for topic clustering. This
workflow uses the openai batch api to reduce costs where 50000 requests can be submitted at once. Despite the 24-hour 
window to run the batch will usually complete in under 3 hours. One oddity of the batch API is that multiple jobs cannot
be queued at once, so jobs must be submitted sequentially.

# Dependencies
The package has the following dependencies:

- numpy
- openai
- sklearn
- pydantic
- torch
- ijson
- faiss
- skopt

# Usage
To install 
```
pip install git+https://github.com/MatthewCorney/clinical-trials.git
```
for usage look in the notebooks folder
