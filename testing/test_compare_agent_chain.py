import pytest
from langchain_core.runnables import  RunnableLambda
import os
import sys

# Get the root directory (parent directory of testing)
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
finbot_path = os.path.join(root_directory, 'finbot')
sys.path.insert(0, finbot_path)

from finbot import rag_chain_agent as rc
from finbot.langsmith_eval import FinbotEval

collection_name = "all_docs_kb_clean_inv"
full_chain = rc.RAGChain(collection_name)
langsmith_eval = FinbotEval()
metadata = langsmith_eval.inspect_collection_upload(collection_name, point_verbose = False)

def capture_and_run_chain(inputs, retriever_chain, main_chain):
    """
    Function to capture documents and invoke the main chain.
    Args:
        inputs (dict): A dictionary containing the inputs for the chain.
        retriever_chain: The retriever chain to be used for document retrieval.
        main_chain: The main chain to be invoked.
    Returns:
        dict: A dictionary containing the final output and retrieved documents.
    """
    # Function to capture documents and invoke the main chain
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

@pytest.fixture
def setup_chain():
    full_chain.create_chains()
    return full_chain

# Test the guided description chain for exposure
def test_simple_compare_perf_chain():
    """
    Guided Exposure Description Prompt:
    Analyze the fund's ({fund_name}), {section} exposure in the context of it's investment objective.
                                                                
    Would you expect this kind of exposure? Explain.
                                                                
    Investment Objective: {inv_obj}  
    """
    # Input to the chain
    input_data = {"question": "Compare the performance between Morgan Stanley Global Fixed Income Opportunities and CT Managed Bond Focused", "top_k": 1, "subject": {"text":"sector"}} #0ZF91
    # Define your specific retriever and main chains
    retriever_chain = full_chain.get_filtered_full_doc_retriever(input_data["question"])
    main_chain = full_chain.comp_simple_perf_chain

    # Call the generalized function
    expected_result = capture_and_run_chain(input_data, retriever_chain, main_chain)

    # Assertions on retrieved documents
    docs = expected_result["retrieved_docs"]
    assert len(docs) == 2, f"More than two document retrieved"
    
    assert "Morgan Stanley Global Fixed Income Opportunities" in docs[0].metadata["fund_name"] or "Morgan Stanley Global Fixed Income Opportunities" in docs[1].metadata["fund_name"], f"Incorrect fund_name retrieved"
    assert "CT Managed Bond Focused" in docs[0].metadata["fund_name"] or "CT Managed Bond Focused" in docs[1].metadata["fund_name"], f"Incorrect fund_name retrieved"

    # Assertions on final output
    result = full_chain.execute_agent_executor(input_data)
    print("Result", result)

    if isinstance(result, dict):
        final_output = result["output"]
    else:
        final_output = result
    assert len(final_output.split()) <= 200, f"Final output is more than 200 words"

# Test the guided description chain for exposure
def test_peer_compare_perf_chain():
    # Input to the chain
    input_data = {"question": "Compare abrdn Global Smaller Companies to its peers"} #0ZF91
    # Define your specific retriever and main chains
    retriever_chain = full_chain.get_filtered_full_doc_retriever(input_data)
    main_chain = full_chain.comp_peer_perf_chain

    # Call the generalized function
    result = capture_and_run_chain(input_data, retriever_chain, main_chain)
    
    # Assertions on retrieved documents
    docs = result["retrieved_docs"]
    #assert len(docs) == 1, f"More than one document retrieved"

    found = False
    for doc in docs:
        if "abrdn Global Smaller Companies" in doc.metadata["fund_name"]:
            found = True
            break
    assert found, f"Incorrect fund_name retrieved"

    # Assertions on final output
    #final_output = result["final_output"]
    result = full_chain.execute_agent_executor(input_data)
    print("Result", result)
    if isinstance(result, dict):
        final_output = result["output"]
    else:
        final_output = result
    assert len(final_output.split()) <= 200, f"Final output is more than 200 words"

# Test the guided description chain for exposure
def test_simple_compare_exp_chain():
    """
    Guided Exposure Description Prompt:
    Analyze the fund's ({fund_name}), {section} exposure in the context of it's investment objective.
                                                                
    Would you expect this kind of exposure? Explain.
                                                                
    Investment Objective: {inv_obj}  
    """
    # Input to the chain
    input_data = {"question": "Compare the asset exposure between Morgan Stanley Global Fixed Income Opportunities and CT Managed Bond Focused", "top_k": 1, "subject": {"text":"sector"}} #0ZF91
    # Define your specific retriever and main chains
    retriever_chain = full_chain.get_filtered_full_doc_retriever(input_data)
    main_chain = full_chain.comp_simple_exp_chain

    # Call the generalized function
    expected_result = capture_and_run_chain(input_data, retriever_chain, main_chain)

    # Assertions on retrieved documents
    docs = expected_result["retrieved_docs"]
    assert len(docs) == 2, f"More than two document retrieved"
    
    assert "Morgan Stanley Global Fixed Income Opportunities" in docs[0].metadata["fund_name"] or "Morgan Stanley Global Fixed Income Opportunities" in docs[1].metadata["fund_name"], f"Incorrect fund_name retrieved"
    assert "CT Managed Bond Focused" in docs[0].metadata["fund_name"] or "CT Managed Bond Focused" in docs[1].metadata["fund_name"], f"Incorrect fund_name retrieved"

    # Assertions on final output
    result = full_chain.execute_agent_executor(input_data)
    print("Result", result)
    if isinstance(result, dict):
        final_output = result["output"]
    else:
        final_output = result
    assert len(final_output.split()) <= 200, f"Final output is more than 200 words"

# Deactivated Tool
# # Test the guided description chain for exposure
# def test_peer_compare_exp_chain():
#     # Input to the chain
#     input_data = {"question": "Compare the sector exposure of abrdn Global Smaller Companies to its peers"} #0ZF91
#     # Define your specific retriever and main chains
#     retriever_chain = full_chain.get_filtered_full_doc_retriever(input_data)
#     main_chain = full_chain.comp_peer_exp_chain

#     # Call the generalized function
#     result = capture_and_run_chain(input_data, retriever_chain, main_chain)
    
#     # Assertions on retrieved documents
#     docs = result["retrieved_docs"]
#     #assert len(docs) == 1, f"More than one document retrieved"

#     found = False
#     for doc in docs:
#         if "abrdn Global Smaller Companies" in doc.metadata["fund_name"]:
#             found = True
#             break
#     assert found, f"Incorrect fund_name retrieved"

#     # Assertions on final output
#     #final_output = result["final_output"]
#     result = full_chain.execute_agent_executor(input_data)
#     print("Result", result)

#     if isinstance(result, dict):
#         final_output = result["output"]
#     else:
#         final_output = result
#     assert len(final_output.split()) <= 200, f"Final output is more than 200 words"