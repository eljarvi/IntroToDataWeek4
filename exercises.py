# %% [markdown]
# # Introduction to Data Science 2025
# 
# # Week 4

# %% [markdown]
# In this week's exercise, we look at prompting and zero- and few-shot task settings. 
# Below is a text generation example from https://github.com/TurkuNLP/intro-to-nlp/blob/master/text_generation_pipeline_example.ipynb 
# demonstrating how to load a text generation pipeline with a pre-trained model and generate text with a given prompt. 
# Your task is to load a similar pre-trained generative model and assess whether the model succeeds at a set of tasks in zero-shot, 
# one-shot, and two-shot settings.
# 
# Note: Downloading and running the pre-trained model locally may take some time. Alternatively, 
# you can open and run this notebook on Google Colab (https://colab.research.google.com/), as assumed in the following example.

# %% [markdown]
# ## Text generation example
# 
# This is a brief example of how to run text generation with a causal language model and pipeline.
# 
# Install transformers (https://huggingface.co/docs/transformers/index) python package. 
# This will be used to load the model and tokenizer and to run generation.

# %%
#!pip install --quiet transformers

# %% [markdown]
# Import the AutoTokenizer, AutoModelForCausalLM, and pipeline classes. 
# The first two support loading tokenizers and generative models from the Hugging Face repository (https://huggingface.co/models), 
# and the last wraps a tokenizer and a model for convenience.

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# %% [markdown]
# Load a generative model and its tokenizer. You can substitute any other generative model name here 
# (e.g. other TurkuNLP GPT-3 models (https://huggingface.co/models?sort=downloads&search=turkunlp%2Fgpt3)), 
# but note that Colab may have issues running larger models. 

# %%
MODEL_NAME = 'TurkuNLP/gpt3-finnish-large'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# %% [markdown]
# Instantiate a text generation pipeline using the tokenizer and model.

# %%
pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    device=model.device
)

# %% [markdown]
# We can now call the pipeline with a text prompt; it will take care of tokenizing, encoding, generation, and decoding:

# %%
output = pipe('Terve, miten menee?', max_new_tokens=25)

print(output)

# %% [markdown]
# Just print the text

# %%
print(output[0]['generated_text'])

# %% [markdown]
# We can also call the pipeline with any arguments that the model "generate" function supports. 
# For details on text generation using transformers, see e.g. this tutorial (https://huggingface.co/blog/how-to-generate).
# 
# Example with sampling and a high temperature parameter to generate more chaotic output:

# %%
output = pipe(
    'Terve, miten menee?',
    do_sample=True,
    temperature=10.0,
    max_new_tokens=25
)

print(output[0]['generated_text'])

# %% [markdown]
# ## Exercise 1
# 
# Your task is to assess whether a generative model succeeds in the following tasks in zero-shot, one-shot, and two-shot settings:
# 
# - binary sentiment classification (positive / negative)
# 
# - person name recognition
# 
# - two-digit addition (e.g. 11 + 22 = 33)
# 
# For example, for assessing whether a generative model can name capital cities, we could use the following prompts:
# 
# - zero-shot:
# 	"""
# 	Identify the capital cities of countries.
# 	
# 	Question: What is the capital of Finland?
# 	Answer:
# 	"""
# - one-shot:
# 	"""
# 	Identify the capital cities of countries.
# 	
# 	Question: What is the capital of Sweden?
# 	Answer: Stockholm
# 	
# 	Question: What is the capital of Finland?
# 	Answer:
# 	"""
# - two-shot:
# 	"""
# 	Identify the capital cities of countries.
# 	
# 	Question: What is the capital of Sweden?
# 	Answer: Stockholm
# 	
# 	Question: What is the capital of Denmark?
# 	Answer: Copenhagen
# 	
# 	Question: What is the capital of Finland?
# 	Answer:
# 	"""
# 
# You can do the tasks either in English or Finnish and use a generative model of your choice from the Hugging Face models repository, for example the following models:
# 
# - English: "gpt2-large"
# - Finnish: "TurkuNLP/gpt3-finnish-large"
# 
# You can either come up with your own instructions for the tasks or use the following:
# 
# - English:
# 	- binary sentiment classification: "Do the following texts express a positive or negative sentiment?"
# 	- person name recognition: "List the person names occurring in the following texts."
# 	- two-digit addition: "This is a first grade math exam."
# - Finnish:
# 	- binary sentiment classification: "Ilmaisevatko seuraavat tekstit positiivista vai negatiivista tunnetta?"
# 	- person name recognition: "Listaa seuraavissa teksteissÃ¤ mainitut henkilÃ¶nnimet."
# 	- two-digit addition: "TÃ¤mÃ¤ on ensimmÃ¤isen luokan matematiikan koe."
# 
# Come up with at least two test cases for each of the three tasks, and come up with your own one- and two-shot examples.

# %%
# Use this cell for your code


#sentiment classification
#zero-shot
output = pipe("""
Ilmaisevatko seuraavat tekstit positiivista vai negatiivista tunnetta?
Vastaa aina vain sanalla "positiivinen" tai "negatiivinen".

Teksti: Onneksi on perjantai.
Vastaus:
""", max_new_tokens=25)
print(output[0]['generated_text'])

#one-shot
output = pipe("""
Ilmaisevatko seuraavat tekstit positiivista vai negatiivista tunnetta?
Vastaa aina vain sanalla "positiivinen" tai "negatiivinen".

Teksti: Onneksi on viikonloppu.
Vastaus: positiivinen

Teksti: Minulla on paha olo.
Vastaus:
""", max_new_tokens=25)
print(output[0]['generated_text'])

#two-shot
output = pipe("""
Ilmaisevatko seuraavat tekstit positiivista vai negatiivista tunnetta?
Vastaa aina vain sanalla "positiivinen" tai "negatiivinen".

Teksti: Sain lahjaksi vain 5 euroa.
Vastaus: negatiivinen

Teksti: Pääsin töistä ajoissa.
Vastaus: positiviinen

Teksti: Koirani on kipeä.
Vastaus:
""", max_new_tokens=25)
print(output[0]['generated_text'])


# finnish exam
#zero-shot
output = pipe("""
Mikä on seuraavien lauseiden predikaatti?

Lause: Koira syö.
Vastaus:
""", max_new_tokens=25)
print(output[0]['generated_text'])

#one-shot
output = pipe("""
Mikä on seuraavien lauseiden predikaatti?

Lause: Minä lähden kauppaan.
Vastaus: lähden

Lause: En jaksa auttaa sinua.
Vastaus:
""", max_new_tokens=25)
print(output[0]['generated_text'])

#two-shot
output = pipe("""
Mikä on seuraavien lauseiden predikaatti?

Lause: Minä lähden kauppaan.
Vastaus: lähden

Lause: En ehdi auttaa sinua.
Vastaus: en ehdi

Lause: Hänellä on paljon kotitehtäviä.
Vastaus:
""", max_new_tokens=25)
print(output[0]['generated_text'])


# food origin
#zero-shot
output = pipe("""
Mistä maasta seuraavat ruoat ovat kotoisin?

Ruoka: sushi
Maa:
""", max_new_tokens=25)
print(output[0]['generated_text'])

#one-shot
output = pipe("""
Mistä maasta seuraavat ruoat ovat kotoisin?

Ruoka: sushi
Maa: Japani

Ruoka: karjalanpiirakka
Maa:
""", max_new_tokens=25)
print(output[0]['generated_text'])

#two-shot
output = pipe("""
Mistä maasta seuraavat ruoat ovat kotoisin?

Ruoka: taco
Maa: Mexico

Ruoka: pizza
Maa: Italia

Ruoka: Gyros
Maa:
""", max_new_tokens=25)
print(output[0]['generated_text'])

# %% [markdown]
# **Submit this exercise by submitting your code and your answers to the above questions as comments on the MOOC platform. You can return this Jupyter notebook (.ipynb) or .py, .R, etc depending on your programming preferences.**

