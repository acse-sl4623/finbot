# finbot

> A package that uses LangChain to query the LLama3 model using customized prompt engineering on a defined investment fund factsheet universe in Qdrant.

The underlying dataset for the domain knowledge is available on One Drive.

https://imperiallondon-my.sharepoint.com/:f:/r/personal/sl4623_ic_ac_uk/Documents/IRP/factsheets/trustnet?csf=1&web=1&e=zQQeM2

This dataset does not need to be downloaded to run the model, as the model uses the defined knowledge bases in Qdrant, API access is ensured through environment configuration.

## Install

```
Only installation of the finbot-env environment is required. 

Use the following commands for conda or pip installation:

Conda:
conda env create -f environment.yml
conda activate finbot-env

Pip:
python -m venv finbot-env
source finbot-env/bin/activate  # On Windows, use `finbot-env\Scripts\activate`
pip install -r requirements.txt

```

## Usage

```
The QA.ipynb notebook provides illustrative questions to the model and allows for interaction.

The LLM will respond by using its domain knowledge or not, based on the query.
The FinBot was also configured to log it's Chain of Thought which will be provided with the response.

The LLama3 model is the base LLM for the FinBot implementation and it runs on the Earth Science and Engineering Department's server, as such connection to the server is required to run the model.

You can use the Imperial VPN to connect to the server.

```

## LangSmith Evaluations

In addition to the pytests in the testing folder which only check for basic functionality, 

LangSmith was used in the evaluation of the model responses based on various Datasets. 

The Datasets mentioned in the IRP Report can be accessed using the following links:

Description Questions - Performance (100 Sample): 
https://smith.langchain.com/public/2b9d8364-88a7-4ad6-b483-f2128b1fe0c6/d

Description Questions - Exposure (100 Sample):
https://smith.langchain.com/public/e115f3f9-b3cb-408e-828d-cd82ad59c7b6/d

General Questions (all types: Docs KB):
https://smith.langchain.com/public/958b93fd-15c9-4bea-9067-6580909f2d91/d

Comparison Questions - Performance (100 Sample)
https://smith.langchain.com/public/d69ab435-6dea-4b81-b29b-eaf0e0d0050e/d

Identification Questions - Performance 
https://smith.langchain.com/public/451f9a3d-b494-4236-a1ec-639d3ff74d63/d

Identification Questions - Exposure
https://smith.langchain.com/public/ff3f5c75-f26f-4833-a112-8f06a6a0a741/d

## Author

**Sara Lakatos**

## License

Copyright © 2024 [Sara Lakatos](sara.lakatos23@imperial.ac.uk)
Licensed under the MIT license.

***
