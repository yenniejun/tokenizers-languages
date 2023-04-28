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
import random

@st.cache_data
def load_data():
	return pd.read_csv('MassiveDatasetValidationData.csv')

def reload_example_text_data():
	random_id = random.choice(val_data['id'])
	tempdf = subset_df[subset_df['id']==random_id]
	tempdf.set_index('lang', inplace=True)
	tempdf = tempdf[['iso', 'text', tokenizer_name]]
	tempdf.columns=['ISO', 'Text', 'Num Tokens']
	tempdf.sort_values(by='ISO', inplace=True)
	st.session_state.examplesdf  = tempdf

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
	st.subheader('Tokenizer')
	# TODO multi-select tokenizers
	tokenizer_name = st.sidebar.selectbox('Select tokenizer', options=tokenizer_names_to_test, label_visibility='collapsed')

	if tokenizer_name not in ['openai/gpt4']:
		url = f'https://huggingface.co/{tokenizer_name}'
		link = f'Tokenizer is available [on the HuggingFace hub]({url})'
		st.markdown(link, unsafe_allow_html=True)
	else:
		link="Tokenized using [tiktoken](https://github.com/openai/tiktoken)"
		st.markdown(link)


	st.subheader('Data')
	with st.spinner('Loading dataset...'):
	    val_data = load_data()
	st.success(f'Data loaded: {len(val_data)}')

	with st.expander('Data Source'):
		st.write("The data in this figure is the validation set of the [Amazon Massive](https://huggingface.co/datasets/AmazonScience/massive/viewer/af-ZA/validation) dataset, which consists of 2033 short sentences and phrases translated into 51 different languages. Learn more about the dataset from [Amazon's blog post](https://www.amazon.science/blog/amazon-releases-51-language-dataset-for-language-understanding)")


	st.subheader('Languages')
	languages = st.multiselect(
		'Select languages',
		options=sorted(val_data.lang.unique()),
		default=['English', 'Spanish' ,'Chinese', 'Burmese'],
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
	
	st.header('Compare tokenization in different languages')
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


	st.subheader('Example Texts')
	
	reload_example_text_data()
	if st.button("🔄 Randomly sample"):
		reload_example_text_data()

	st.dataframe(st.session_state.examplesdf)  # Same as st.write(df)


	






	with st.expander("About the project"):
		st.write("The purpose of this project is to compare the tokenization length for different languages. For some tokenizers, tokenizing a message in one language may result in 10-20x more tokens than a comparable message in another language (e.g. try English vs. Burmese). This is part of a larger project of measuring inequality in NLP.")

