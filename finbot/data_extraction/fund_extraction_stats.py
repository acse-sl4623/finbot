import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
#print("PYTHONPATH:", sys.path)

from finbot.rag_chain import RAGChain
from finbot.evaluations.langsmith_eval import FinbotEval

langsmith_eval = FinbotEval()
collection_name = "all_docs_kb_clean_inv"
rag_chain = RAGChain(collection_name)

# Load the vector store associated to the RAGChain
vectorstore = rag_chain.loaded_vectorstore
rag_chain.create_chains()  # Initialize all the chains

# Extract the unique metadata values from the collection
metadata = langsmith_eval.inspect_collection_upload(collection_name, point_verbose = False)

print("Number of unique asset classes:", len(metadata['asset_class']))
print("Unique asset classes:", metadata['asset_class'])

print("Number of fund names:", len(metadata['fund_name']))
#print("Unique fund names:", metadata['fund_name'])

print(f"Number of unique launch dates: {len(metadata['launch_date'])} ({len(metadata['launch_date'])/len(metadata['fund_name'])*100})")
print(f"Number of unique ISINs: {len(metadata['ISIN'])} ({len(metadata['ISIN'])/len(metadata['fund_name'])*100})")
print(f"Number of unique SEDOLs: {len(metadata['SEDOL'])} ({len(metadata['SEDOL'])/len(metadata['fund_name'])*100})")
print(f"Number of unique investment objectives: {len(metadata['inv_obj'])} ({len(metadata['inv_obj'])/len(metadata['fund_name'])*100})")