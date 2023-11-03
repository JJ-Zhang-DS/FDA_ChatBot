# FDA_STAT_ChatBot 
```
This is a ChatBot for the project of NLP class using streamlit 
```

The aim is to help you get FDA guidline for the statistical requirement during drug discoveries. 

The building procedure follows this [website](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/).

## Usage
- You should have a valid conda/python environment to begin with.
```
conda create -n chatbot python=3.10
conda activate chatbot
```

- Download the repo
```
git clone git@github.com:JJ-Zhang-DS/FDA_ChatBot.git
cd FDA_ChatBot
```

- Install the dependencies
```
pip install -r requirements.txt
```

- Testcase: with user's data
	* Note: you need to get your openAI API key ready at this step and store it in the file `secrets.toml` under `.streamlit` directory.

```
cd with_data
touch .streamlit/secrets.toml
```

Use your favorite text editor to modify the file `secrets.toml` under `.streamlit` directory. It should contain the infomation below:

```
openai_key = "<your own openai api key>"
```

	* Run on your local server
```
streamlit run streamlit_app.py
```

- Testcase: without user's data
	* Note: you need to get your openAI API key ready at this step and store it in the file `secrets.toml` under `.streamlit` directory.

```
cd wo_data
touch .streamlit/secrets.toml
```

Use your favorite text editor to modify the file `secrets.toml` under `.streamlit` directory. It should contain the infomation below:

```
openai_key = "<your own openai api key>"
```

	* Run on your local server
```
streamlit run streamlit_app.py
```
