import streamlit as st
from collections import defaultdict
import tqdm
import transformers
from transformers import AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px

tokenizer_names_to_test = [
  "openai/gpt4",
  "xlm-roberta-base",  # old style
  "bert-base-uncased",  # old style
  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "bigscience/bloom",  # HuggingFace
  "StabilityAI/stablelm-base-alpha-7b",  # StableLM with Open Assistant
  "google/flan-t5-base",  # Flan T5 (better than T5), Google
  "facebook/mbart-large-50",  # Facebook
  "facebook/nllb-200-distilled-600M",  # Facebook
  "EleutherAI/gpt-neox-20b",  # same as Pythia
]

with st.sidebar:
	with st.spinner('Loading dataset...'):
	    val_data = pd.read_csv('MassiveDatasetValidationData.csv')
	st.success(f'Data loaded: {len(val_data)}')

	languages = st.multiselect(
		'Select languages',
		options=sorted(val_data.lang.unique()),
		default=['English', 'Spanish' ,'Chinese'],
		max_selections=5
	)

	# TODO multi-select tokenizers
	# TODO add openai to this options
	tokenizer_name = st.sidebar.selectbox('Tokenizers', options=tokenizer_names_to_test)
	st.write('You selected:', tokenizer_name)

	# with st.spinner('Loading tokenizer...'):
	#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
	# st.success(f'Tokenizer loaded: {tokenizer_name}')

	# # TODO - preload the tokenized versions ... much easier!
	# # TODO - add the metadata data as well??? later on maybe
	# with st.spinner('Calculating tokenization for data...'):
	# 	if tokenizer_name not in val_data.columns:
	# 		val_data[f'{tokenizer_name}'] = val_data.text.apply(lambda x: len(tokenizer.encode(x)))
	# st.success('Completed.')

with st.container():
	if tokenizer_name in val_data.columns:
		subset_df = val_data[val_data.lang.isin(languages)]
		subset_data = [val_data[val_data.lang==_lang][tokenizer_name] for _lang in languages]

		
	fig = ff.create_distplot(subset_data, group_labels=languages, show_hist=False)
	st.plotly_chart(fig, use_container_width=True)


	# for _lang in languages:
	# 	subset = val_data[val_data.lang==_lang]

	# 	fig = ff.create_distplot(val_data,  bin_size=.5,
	#                          curve_type='normal', # override default 'kde'
	#                          colors=colors)



