import pytest
from fuzzywuzzy import fuzz

from finbot import rag_chain_agent as rc
from finbot.langsmith_eval import FinbotEval

langsmith_eval = FinbotEval()
collection_name = "all_sec_kb_clean_inv"
test_chain = rc.RAGChain(collection_name)
metadata = langsmith_eval.inspect_collection_upload("all_docs_kb_clean_inv") 
fund_names = metadata["fund_name"]

@pytest.fixture
def setup_chain():
    test_chain.create_chains()
    return test_chain


@pytest.mark.parametrize(
    "inputs, expected_response",
    [
        (
            # 0GF7R
            {"question": "What's the highest sector allocation of HSBC Global Strategy Dynamic?"},
            ["information technology", "17"]
        ),
        (
            {"question": "What is the 1 year return of HSBC Global Strategy Dynamic Portfolio"},
            ["14"]
        ),
        (
            {"question": "What is the investment objective of Fidelity Multi Asset Allocator Defensive W Acc"},  # 0DFT8
            []
        ),
        (
            {"question": "How many investment funds are you aware of?"},
            ["799"]
        ),
        (
            {"question": "What pizza should i have tonight?"},
            []
        )
    ]
)
def test_general_chain_output_through_agent(setup_chain, inputs, expected_response):
    # Use the fixture to get the initialized chain
    test_chain = setup_chain

    # Run the agent execution chain
    # Answer is a dictionary of question and answer
    result = test_chain.execute_agent_executor(inputs)
    result = result["output"]

    print(f"Question: {inputs['question']}")
    print("Answer: ", result, type(result))

    # Verify that the result is as expected
    assert result is not None, "Result is None"

    # Check if the result contains the expected substrings
    for expected in expected_response:
        assert expected.lower() in result.lower(), f"Expected '{expected}' not found in result."

def test_list_general_knowledge_through_agent(setup_chain):
    # Use the fixture to get the initialized chain
    test_chain = setup_chain
    inputs = {"question": "List 10 investment funds from the knowledge base"}
    # Run the agent execution chain
    # Answer is a dictionary of question and answer
    result = test_chain.execute_agent_executor(inputs)
    result = result["output"]

    # Check that 10 fund names are in the string
    # Tokenize the string and convert to lowercase
    tokens = result.split('\n')
    #print(tokens)
    found_funds = []
    for token in tokens:
        for fund_name in fund_names:
            if fuzz.ratio(token.lower(), fund_name.lower()) >= 80: # Use fuzzy matching with >80% similarity
                found_funds.append(fund_name)
                break  # Stop checking this token if a match is found
    
    assert len(found_funds) == 10, f"Found {len(found_funds)} fund names in the output: {found_funds}"
