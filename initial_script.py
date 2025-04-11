# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 23:11:35 2025

@author: johny
"""

from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd
import requests
import seaborn as sns
import tiktoken
from sklearn.model_selection import train_test_split
#from tqdm.auto import tqdm
#from typing import Optional
#name: Optional[str] = None
from __future__ import annotations


LLM_SERVER = "http://localhost:11434"
MODEL = "gemma3:1b"
DATA_DIR = Path.cwd().parent / "data" # C:/users/johny/data

# need to download training data from drivendata...
df = pd.read_csv( "F:/Data_Science_projects/llm_drivendata/train.csv", index_col=0)
df

# paper_id: identifier of the paper provided by the Open Science Framework Preprints server
# from the Center for Open Science

# text: The main body of the academic paper. This is a modified version of the original
# to remove the abstract, references, and other text that would make it into an abstract

# abstract: The paper abstract that was provided along with the paper's preprint

# inspect a couple 

print(f"Document length: {len(df.loc[0, 'text']):,} characters")
print("Document:")
print(fill(df.loc[0,'text'],replace_whitespace=False)[:1000]) # first 1000 lines
print(fill(df.loc[0,'summary']))
len(df.loc[0,'summary']) # 1302 characters

# let's get a sense of the size of the test dataframe
df["text_len"] = df.text.str.len()
df["summary_len"] = df.summary.str.len()
df["text_len_log"] = np.log10(df.text_len)
df["summary_len_log"] = np.log10(df.summary_len)

# plot the log results
g = sns.jointplot(df, x="text_len_log", y="summary_len_log", kind="hist")
xticks = [3,4,5,6]
yticks = [2, np.log10(250), np.log10(500), 3, np.log10(2_500), np.log10(5_000),4]
g.ax_marg_x.set_xticks(xticks, [f"{10**tick:,.0f}" for tick in xticks])          
g.ax_marg_y.set_yticks(yticks, [f"{10**tick:,.0f}" for tick in yticks])
g.set_axis_labels("Num. characters in text (log scale)", "Num. characters in summary (log scale)")
print(g)

# further investigation of df
df.describe()

#%% Split the data
# split the data
train, test = train_test_split(df, test_size=0.3, random_state=0)
f"Train shape: {train.shape}; Test shape: {test.shape}"

# split again!
train, validation = train_test_split(train, test_size=0.3, random_state=0)
f"Train shape: {train.shape}, Validation shape: {validation.shape}, Test shape: {test.shape}"
# still have our test of 300 samples but have split our train further to train (490) and validation (210) subsets

# Need to implement exhaustive validation techniques: leave-one-out LOOCV or leave-p-out LpOCV
#  see: https://en.wikipedia.org/wiki/Cross-validation_(statistics)

#%% Prompting

def doc_and_summary_from_row(doc_row: pd.Series):
    """Pull out the document and summary"""
    return doc_row["text"], doc_row["summary"]

doc_row = train.loc[533]

def show_doc_and_summary(doc: str, summary: str, max_len_to_print: int=500) -> str:
    """Show a little bit of doc and its summary"""
    
    return (
    
       f"Document ({max_len_to_print:,} of {len(doc):,} characters):\n"
        f"{fill(doc[:max_len_to_print], replace_whitespace=False)}...\n\n"
        f"Summary ({len(summary):,} characters):\n"
        f"{fill(summary)}"
    )
    
doc, summary = doc_and_summary_from_row(doc_row)
print(show_doc_and_summary(doc, summary))

# let's make our prompt..
prompt_template = 'Here is a terrific one-sentence summary of "{doc}": '
prompt = prompt_template.format(doc=doc)


def show_prompt(prompt: str, start_chars: int = 60, end_chars: int = 30) -> str:
    """Nicely format a prompt"""
    return f"Our prompt ({len(prompt):,} characters):\n{prompt[:start_chars]} ... {prompt[-end_chars:]}"


print(show_prompt(prompt))

# lets see if our LLM is running
requests.get(LLM_SERVER + "/api/version").json()
# no error returned so looking good!

# Let's call Ollama completion API...

def get_llm_completion(
    prompt: str, max_tokens: int | None = None, top_k: int | None = None) -> str:
    """Hit an API endpoint to get an LLM completion"""
    data = {
        "model": MODEL,
        "prompt": prompt,
        "seed": 0,
        "max_tokens": max_tokens,
        "top_k": top_k,
    }
    resp = requests.post(LLM_SERVER + "/v1/completions", json=data)
    return resp.json()["choices"][0]["text"]

completion = get_llm_completion(prompt)
print(prompt_template + "\n\n" + fill(completion, replace_whitespace=False)[:1000])

# How many tokens is the size of our prompt? Ollama has a limit of 2048 tokens

def count_tokens(text: str) -> int:
    """Count the number of tokens in a string"""
    enc = tiktoken.encoding_for_model("gpt-4o")
    return len(enc.encode(text))

print(f"Our prompt is {count_tokens(prompt):,} tokens.")
# which is a lot larger than 2048 tokens!

# Make the prompt itself shorter by simply only selecting the 'front' part of it hoping 
# doesn't lose too much...

shorter_prompt = prompt_template.format(doc=doc[:7500])
print(f"Our shorter prompt is {count_tokens(shorter_prompt):,} tokens long.")
print(show_prompt(shorter_prompt))

# let's try the summary of this shorter prompt
completion = get_llm_completion(shorter_prompt)
print(fill(completion, replace_whitespace=False)[:1000])

# Need to next address the fact our model is 'instruction-tuned' as still don't have a sensible output

