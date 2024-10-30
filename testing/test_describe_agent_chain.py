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
def test_guided_description_exposure_chain():
    """
    Guided Exposure Description Prompt:
    Analyze the fund's ({fund_name}), {section} exposure in the context of it's investment objective.
                                                                
    Would you expect this kind of exposure? Explain.
                                                                
    Investment Objective: {inv_obj}  
    """
    # Input to the chain
    input_data = {"question": "Describe the sector allocation of Schroder Strategic Bond?", "top_k": 1, "subject": {"text":"sector"}} #0ZF91
    # Define your specific retriever and main chains
    retriever_chain = full_chain.get_filtered_full_doc_retriever(input_data["question"])
    main_chain = full_chain.description_exp_chain

    # Call the generalized function
    expected_result = capture_and_run_chain(input_data, retriever_chain, main_chain)

    # Assertions on retrieved documents
    docs = expected_result["retrieved_docs"]
    assert len(docs) == 1, f"More than one document retrieved"
    
    doc = docs[0]
    assert "Schroder Strategic Bond" in doc.metadata["fund_name"], f"Incorrect fund_name retrieved"

    # Assertions on final output
    result = full_chain.execute_agent_executor(input_data)
    final_output = result["output"] #expected_result["final_output"]
    assert len(final_output.split()) <= 200, f"Final output is more than 200 words"

# Test the guided description chain for exposure
def test_guided_description_performance_chain():
    """
    Guided Performance Description Prompt:
    1. Identify notable changes in the historical performance of the fund over the various time periods provided.
    2. Identify if the fund has generally outperformed or underperformed its benchmark (if the benchmark exists), notably in the last 1 year
    3. Identify notable changes in the rank within sector and quartile rank over the time periods provided, which are indicative of the
    the fund's performance compared to its peers.
    4. Compute for the 1 year return:
        a) active performance of the fund compared to its benchmark over the time periods provided
        b) percentile rank of the fund compared to its peers over the time periods provided
    """
    # Input to the chain
    input_data = {"question": "Describe the performance of Schroder Strategic Bond?"} #0ZF91
    # Define your specific retriever and main chains
    retriever_chain = full_chain.get_filtered_full_doc_retriever(input_data)
    main_chain = full_chain.description_perf_chain

    # Call the generalized function
    result = capture_and_run_chain(input_data, retriever_chain, main_chain)
    
    # Assertions on retrieved documents
    docs = result["retrieved_docs"]
    assert len(docs) == 1, f"More than one document retrieved"

    doc = docs[0]
    assert "Schroder Strategic Bond" in doc.metadata["fund_name"], f"Incorrect fund_name retrieved"

    # Assertions on final output
    #final_output = result["final_output"]
    result = full_chain.execute_agent_executor(input_data)
    final_output = result["output"]

    # Determine if the fund outperformed or undeperformed its benchmark
    active_perf_1_year = float(doc.metadata["Cumulative Performance"]["1yr"]["Fund"]) - float(doc.metadata["Cumulative Performance"]["1yr"]["Benchmark"])
    if active_perf_1_year > 0:
        assert "outperform" in final_output, f"Did not identify 1yr outperformance"
    else:
        assert "underperform" in final_output, f"Did not identify 1yr underperformance"
    
    # Determine the percentile rank of the fund compared to its peers
    result = full_chain.execute_agent_executor(input_data)
    final_output = result["output"] #expected_result["final_output"]
    assert len(final_output.split()) <= 200, f"Final output is more than 200 words"