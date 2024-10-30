import pytest

from finbot import rag_chain_agent as rc

collection_name = "all_docs_kb_clean_inv"
full_chain = rc.RAGChain(collection_name)

@pytest.fixture
def setup_chain():
    full_chain.create_chains()
    return full_chain

def test_subject_classification_chain_input(setup_chain):
    """
    Test case for subject_classification_chain.invoke() method.
    Args:
        setup_chain: The fixture to get the initialized chain.
    Raises:
        AssertionError: If the result is not as expected.
    ChatGBT: https://chatgpt.com/share/a66aca6f-9f73-4caa-bb0e-e8da29dac8fc
    """
    full_chain = setup_chain  # Use the fixture to get the initialized chain

    # Define the input question
    question = "Compare the sector exposure of FTF Martin Currie UK Equity Income and FP Octopus UK Future Generations"
    
    # Run the extract_fund_names_chain with the question
    result = full_chain.subject_classification_chain.invoke(question)
    print("Subject: ", result, type(result)) # dictionary object with key "text" and value string

    # Verify that the result is as expected
    assert result is not None
    expected_fund_names = ["sector"]
    
    # Check if the result contains the expected fund names
    for subject in expected_fund_names:
        assert subject in result['text'].lower()

def test_extract_fund_names_chain_input(setup_chain):
    """
    Test case for the extract_fund_names_chain function.
    Args:
        setup_chain: The initialized chain fixture.
    Raises:
        AssertionError: If the result is not as expected.
    """

    full_chain = setup_chain  # Use the fixture to get the initialized chain

    # Define the input question
    question = "Compare the sector exposure of FTF Martin Currie UK Equity Income and FP Octopus UK Future Generations"
    
    # Run the extract_fund_names_chain with the question
    result = full_chain.extract_fund_names_chain.invoke(question)
    print("Fund Names: ", result, type(result))

    # Verify that the result is as expected
    assert result is not None # the output is a string not a list object
    expected_fund_names = ["FTF Martin Currie UK Equity Income", "FP Octopus UK Future Generations"]
    
    # Check if the result contains the expected fund names
    for fund in expected_fund_names:
        assert fund.lower() in result['text'].lower()