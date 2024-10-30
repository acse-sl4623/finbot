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

#langsmith_eval.ls_client.delete_dataset(dataset_id="6fec4c21-9b46-474a-a8ad-b94fe6f87087")
# Create a dataset for the Investment Objective prompts
dataset_name = "General Questions (all types: Docs KB)"
dataset_description = "General questions on data prompts"

datasets = langsmith_eval.ls_client.list_datasets()  # List all datasets
existing_dataset = next((ds for ds in datasets if ds.name == dataset_name), None)

# Create new dataset
if existing_dataset:
    dataset = existing_dataset
    print(f"Dataset '{dataset_name}' already exists. Loaded existing dataset.")
else:
    dataset = langsmith_eval.ls_client.create_dataset(
                    dataset_name=dataset_name,
                    description=dataset_description
                )    

    # Specific question: investment objective
    list_names = metadata['fund_name']
    random_fund_names = random.sample(list_names, 20)
    for name in random_fund_names:
        for input_query in ["What is the investment objective of " + name + "?"]:
            input_data = {"question": input_query}

            # Retrieve "ground truth" as 
            retriever_chain = rag_chain.get_filtered_full_doc_retriever(input_data["question"])
            main_chain = rag_chain.domain_specific_general_chain
            expected_result = capture_and_run_chain(input_data, retriever_chain, main_chain)
            print("Captured document retrieval")

            # Extract retrieved documents from expected run
            docs = expected_result["retrieved_docs"]
            print("Number of documents retrieved: ", len(docs))
            # assert len(docs) == 1, f"More than one document retrieved"
            # doc = docs[0]

            # Build dataset point with context as "ground truth"
            context_str = rag_chain.gen_build_context(docs) # it's a string
            langsmith_eval.ls_client.create_example(
                inputs={"question": input_query},
                outputs={"context": context_str},
                dataset_id=dataset.id
            )

    for asset in metadata['asset_class']:
        for input_query in [f"Name {random.randint(1, 10)} {asset} fund from the knowledge base."]:
            input_data = {"question": input_query}
            # Retrieve "ground truth" as 
            retriever_chain = rag_chain.get_filtered_full_doc_retriever(input_data["question"])
            main_chain = rag_chain.domain_specific_general_chain
            expected_result = capture_and_run_chain(input_data, retriever_chain, main_chain)
            print("Captured document retrieval")

            # Extract retrieved documents from expected run
            docs = expected_result["retrieved_docs"]
            print("Number of documents retrieved: ", len(docs))
            # assert len(docs) == 1, f"More than one document retrieved"
            # doc = docs[0]

            # Build dataset point with context as "ground truth"
            context_str = rag_chain.gen_build_context(docs) # it's a string
            langsmith_eval.ls_client.create_example(
                inputs={"question": input_query},
                outputs={"context": context_str},
                dataset_id=dataset.id
            )
    
    # General question: on a particular asset
    for asset in metadata['asset_class']:
        for input_query in [f"What is the investment objective of a {asset} fund?"]:
            input_data = {"question": input_query}

            # Retrieve "ground truth" as 
            retriever_chain = rag_chain.get_filtered_full_doc_retriever(input_data["question"])
            main_chain = rag_chain.domain_specific_general_chain
            expected_result = capture_and_run_chain(input_data, retriever_chain, main_chain)
            print("Captured document retrieval")

            # Extract retrieved documents from expected run
            docs = expected_result["retrieved_docs"]
            print("Number of documents retrieved: ", len(docs))
            # assert len(docs) == 1, f"More than one document retrieved"
            # doc = docs[0]

            # Build dataset point with context as "ground truth"
            context_str = rag_chain.gen_build_context(docs) # it's a string
            langsmith_eval.ls_client.create_example(
                inputs={"question": input_query},
                outputs={"context": context_str},
                dataset_id=dataset.id
            ) 
    print(f"Dataset '{dataset_name}' created successfully.")

    # Identify specific performance data
    time_periods = ["3 month", "6 month", "1 year", "3 year", "5 year"]
    performance_type = ["cumulative", "discrete", ""]
    for asset in metadata['asset_class']:
        random_period = random.choice(time_periods)
        random_performance = random.choice(performance_type)
        for input_query in [f"What is the {random_period} {random_performance} return of a {asset} fund?"]:
            print("Question: ", input_query)
            input_data = {"question": input_query}
            # Retrieve "ground truth" as 
            retriever_chain = rag_chain.get_filtered_full_doc_retriever(input_data["question"])
            main_chain = rag_chain.domain_specific_general_chain
            expected_result = capture_and_run_chain(input_data, retriever_chain, main_chain)
            print("Captured document retrieval")

            # Extract retrieved documents from expected run
            docs = expected_result["retrieved_docs"]
            print("Number of documents retrieved: ", len(docs))
            # assert len(docs) == 1, f"More than one document retrieved"
            # doc = docs[0]

            # Build dataset point with context as "ground truth"
            context_str = rag_chain.gen_build_context(docs) # it's a string
            langsmith_eval.ls_client.create_example(
                inputs={"question": input_query},
                outputs={"context": context_str},
                dataset_id=dataset.id
            )

    # Identify section
    list_names = metadata['fund_name']
    random_fund_names = random.sample(list_names, 20)
    # Identify section
    sections = ['regional', 'sector', 'issuer']
    request_types = ["highest", "lowest", "two highest", "two lowest", "top", "bottom"]
    for fund_name in random_fund_names:
        sec = random.choice(sections)
        request_type = random.choice(request_types)
        for input_query in [f"What's the {request_type} {sec} allocation of {fund_name}?"]:
            input_data = {"question": input_query}

            # Retrieve documents relevant to query
            retriever_chain = rag_chain.get_filtered_full_doc_retriever(input_data["question"])
            main_chain = rag_chain.domain_specific_general_chain
            expected_result = capture_and_run_chain(input_data, retriever_chain, main_chain)
            print("Captured document retrieval")

            # Extract retrieved documents from expected run
            docs = expected_result["retrieved_docs"]
            print("Number of documents retrieved: ", len(docs))
            # assert len(docs) == 1, f"More than one document retrieved"
            # doc = docs[0]

            # Build dataset point with context as "ground truth"
            context_str = rag_chain.gen_build_context(docs) # it's a string
            langsmith_eval.ls_client.create_example(
                inputs={"question": input_query},
                outputs={"context": context_str},
                dataset_id=dataset.id
            )
        
    # Identify section
    sections = ['regional', 'sector', 'issuer']
    for sec in sections:
        for input_query in [f"What's the lowest {sec} allocation of a Fixed Interest fund?"]:
            input_data = {"question": input_query}

            # Retrieve documents relevant to query
            retriever_chain = rag_chain.get_filtered_full_doc_retriever(input_data["question"])
            main_chain = rag_chain.domain_specific_general_chain
            expected_result = capture_and_run_chain(input_data, retriever_chain, main_chain)
            print("Captured document retrieval")

            # Extract retrieved documents from expected run
            docs = expected_result["retrieved_docs"]
            print("Number of documents retrieved: ", len(docs))
            # assert len(docs) == 1, f"More than one document retrieved"
            # doc = docs[0]

            # Build dataset point with context as "ground truth"
            context_str = rag_chain.gen_build_context(docs) # it's a string
            langsmith_eval.ls_client.create_example(
                inputs={"question": input_query},
                outputs={"context": context_str},
                dataset_id=dataset.id
            )

    # Test List function
    list_values = ['5', '20', '100', 'all']
    for list_value in list_values:
        for input_query in [f"List {list_value} investment funds"]:
            input_data = {"question": input_query}

            # Retrieve documents relevant to query
            retriever_chain = rag_chain.get_filtered_full_doc_retriever(input_data["question"])
            main_chain = rag_chain.list_funds_in_knowledge_base_chain
            expected_result = capture_and_run_chain(input_data, retriever_chain, main_chain)
            print("Captured document retrieval")

            # Extract retrieved documents from expected run
            docs = expected_result["retrieved_docs"]
            print("Number of documents retrieved: ", len(docs))

            # Build dataset point with context as "ground truth"
            context_str = rag_chain.gen_build_list_context(docs)
            langsmith_eval.ls_client.create_example(
                inputs={"question": input_query},
                outputs={"context": context_str},
                dataset_id=dataset.id
            )
        
    # Count
    input_data = {"question": "How many investment funds are you aware of?"}
    # Build dataset point with context as "ground truth"
    langsmith_eval.ls_client.create_example(
        inputs={"question": input_query},
        outputs={"context": '799'},
        dataset_id=dataset.id
    )
                
langsmith_eval.run_evaluation_on_dataset(dataset_name, rag_chain.execute_simple_llm, run_name = "simple_chain_gen_understanding")
langsmith_eval.run_evaluation_on_dataset(dataset_name, rag_chain.execute_domain_specific_chain, run_name = "domain_chain_gen_understanding")
langsmith_eval.run_evaluation_on_dataset(dataset_name, route_chain.execute_guided_full_chain, run_name = "route_chain_gen_understanding")
langsmith_eval.run_evaluation_on_dataset(dataset_name, rag_chain.execute_agent_executor, run_name = "agent_chain_gen_understanding")