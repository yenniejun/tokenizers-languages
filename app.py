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

@st.cache_data
def load_data():
	return pd.read_csv('MassiveDatasetValidationData.csv')

# TODO allow new tokenizers from HF
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
	st.subheader('Model')
	# TODO multi-select tokenizers
	tokenizer_name = st.sidebar.selectbox('Select tokenizer', options=tokenizer_names_to_test)

	st.subheader('Data')
	with st.spinner('Loading dataset...'):
	    val_data = load_data()
	st.success(f'Data loaded: {len(val_data)}')

	languages = st.multiselect(
		'Select languages',
		options=sorted(val_data.lang.unique()),
		default=['English', 'Spanish' ,'Chinese'],
		max_selections=6,
		label_visibility='collapsed'
	)
	
	st.subheader('Figure')
	show_hist = st.checkbox('Show histogram', value=False)
	# dist_marginal = st.radio('Select distribution', options=['box', 'violin', 'rug'], horizontal=True)

	# with st.spinner('Loading tokenizer...'):
	#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
	# st.success(f'Tokenizer loaded: {tokenizer_name}')

	# # TODO - add the metadata data as well??? later on maybe
	# with st.spinner('Calculating tokenization for data...'):
	# 	if tokenizer_name not in val_data.columns:
	# 		val_data[f'{tokenizer_name}'] = val_data.text.apply(lambda x: len(tokenizer.encode(x)))
	# st.success('Completed.')

with st.container():
	if tokenizer_name in val_data.columns:
		subset_df = val_data[val_data.lang.isin(languages)]
		subset_data = [val_data[val_data.lang==_lang][tokenizer_name] for _lang in languages]
	
	st.header('Tokenization in different languages')
	st.divider()
	fig = ff.create_distplot(subset_data, group_labels=languages, show_hist=show_hist)

	fig.update_layout(
		title=dict(text=tokenizer_name, font=dict(size=25), automargin=True, yref='paper', ),
		# title=tokenizer_name,
		xaxis_title="Number of Tokens",
    yaxis_title="Density",
    # title_font_family='"Source Sans Pro", sans-serif'
	) 
	st.plotly_chart(fig, use_container_width=True)

	st.subheader('Median Token Length')
	metric_cols = st.columns(len(languages))
	for i, _lang in enumerate(languages):
		metric_cols[i].metric(_lang, int(np.median(subset_df[subset_df.lang==_lang][tokenizer_name])))

	if tokenizer_name not in ['openai/gpt4']:
		url = f'https://huggingface.co/{tokenizer_name}'
		link = f'[Find on the HuggingFace hub]({url})'
		st.markdown(link, unsafe_allow_html=True)

