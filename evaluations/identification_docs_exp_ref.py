import sys
import os
import random

from langchain_core.runnables import RunnableLambda
from langsmith.schemas import Example, Run
from langsmith.evaluation import evaluate

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


from finbot.rag_chain_agent import RAGChain
from finbot.rag_chain_route import RAGChainRoute
from finbot.langsmith_eval import FinbotEval

langsmith_eval = FinbotEval()

import os

# Disable parallelism for huggingface/tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

collection_name = "all_docs_kb_clean_inv"
rag_chain = RAGChain(collection_name)
route_chain = RAGChainRoute(collection_name)

# Load the vector store associated to the RAGChain
vectorstore = rag_chain.loaded_vectorstore
rag_chain.create_chains()  # Initialize all the chains

# Extract the unique metadata values from the collection
metadata = langsmith_eval.inspect_collection_upload(collection_name, point_verbose = False)

# Capture the retrieved documents for a "GOOD" run
def capture_and_run_chain(inputs, retriever_chain, main_chain):
    # Capture the retrieval documents by wrapping the general chain
    def capture_documents(inputs):
        # Extract the question and top_k values from the inputs dictionary
        question = inputs["question"]

        # Capture the documents retrieved during the invocation
        # Use the passed retriever_chain with the float value of top_k and the string value of question
        documents = retriever_chain.invoke(question)

        # Use the passed main_chain and invoke it with the inputs dictionary
        final_output = main_chain.invoke(question)

        # Return both final output and retrieved documents
        return {
            "final_output": final_output,
            "retrieved_docs": documents
        }

    # Wrap in a RunnableLambda to integrate with your chain execution
    wrapped_chain = RunnableLambda(capture_documents)
    return wrapped_chain.invoke(inputs)

# Create new dataset
# Create a dataset for the Investment Objective prompts
#langsmith_eval.ls_client.delete_dataset(dataset_id="d510dbe4-6135-4a1a-836a-ed2cd9663fff")
dataset_name = "Identification Questions - Exposure"
dataset_description = "identification prompts about exposure"

datasets = langsmith_eval.ls_client.list_datasets()  # List all datasets
existing_dataset = next((ds for ds in datasets if ds.name == dataset_name), None)

if existing_dataset:
    dataset = existing_dataset
    print(f"Dataset '{dataset_name}' already exists. Loaded existing dataset.")
else:
    dataset = langsmith_eval.ls_client.create_dataset(
                    dataset_name=dataset_name,
                    description=dataset_description
                )
    
    queries_top_perf = [
                    "Describe the sector allocation of the fund with the best performance in the Asia peer group",
                    "Describe the sector allocation of the fund with the best performance in the Growth peer group",
                    "Describe the sector allocation of the fund with the best performance in the Hedge Fund peer group",
                    "Describe the sector allocation of the fund with the best performance in the India peer group",
                    "Describe the sector allocation of the fund with the best performance in the UK Gilts peer group",
    ]

    queries_bottom_perf = ["Describe the sector allocation of the fund with the worst performance in the Total Return peer group.",
                    "Describe the sector allocation of the fund with the worst performance in the Japan peer group",
                    "Describe the sector allocation of the fund with the worst performance in the Income peer group",
                    "Describe the sector allocation of the fund with the worst performance in the US Technology peer group",
                    "Describe the sector allocation of the fund with the worst performance in the High Yield peer group",

    ]
    
    # Top Performers
    for input_query in queries_top_perf:
        # Get peer group and outlier
        peer_group = rag_chain.create_peer_group(input_query)
        top_performer = rag_chain.identify_top_performer(input_query)
        context_str = f"The top performer in peer group is {top_performer}\n"
        for fund_name, fund_rank in peer_group.items():
            context_str += f"{fund_name} has rank {fund_rank['rank']}\n"

        # Get Document of outlier
        input_data = {"question": top_performer, "top_k":1, "subject": {"text": "sector"}}
        retriever_chain = rag_chain.get_filtered_full_doc_retriever(input_data["question"])
        main_chain = rag_chain.description_exp_chain
        expected_result = capture_and_run_chain(input_data, retriever_chain, main_chain)
        doc_top_performer = expected_result["retrieved_docs"]
        context_perf = rag_chain.build_exp_context(doc_top_performer, {"text" : "sector"}) # it's a string
        
        # Pass context as reference
        context_str += context_perf
        langsmith_eval.ls_client.create_example(
            inputs={"question": input_query},
            outputs={"context": context_str},
            dataset_id=dataset.id
        )

    # Worst Performers
    for input_query in queries_bottom_perf:
        peer_group = rag_chain.create_peer_group(input_query)
        top_performer = rag_chain.identify_worst_performer(input_query)
        context_str = f"The worst performer in peer group is {top_performer}\n"
        for fund_name, fund_rank in peer_group.items():
            context_str += f"{fund_name} has rank {fund_rank['rank']}\n"
        
        # Get Document of outlier
        input_data = {"question": top_performer, "top_k":1, "subject": {"text": "sector"}}
        retriever_chain = rag_chain.get_filtered_full_doc_retriever(input_data["question"])
        main_chain = rag_chain.description_exp_chain
        expected_result = capture_and_run_chain(input_data, retriever_chain, main_chain)
        doc_top_performer = expected_result["retrieved_docs"]
        context_perf = rag_chain.build_exp_context(doc_top_performer, {"text" : "sector"}) # it's a string
        
        # Pass context as reference
        context_str += context_perf
        langsmith_eval.ls_client.create_example(
            inputs={"question": input_query},
            outputs={"context": context_str},
            dataset_id=dataset.id
        )
#Run the evaluation on the dataset  
langsmith_eval.run_evaluation_on_dataset(dataset_name, rag_chain.execute_domain_specific_chain, run_name = "domain_chain_id_exp")
langsmith_eval.run_evaluation_on_dataset(dataset_name, rag_chain.execute_agent_executor, run_name = "agent_chain_id_exp")