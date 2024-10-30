import pytest
from fuzzywuzzy import fuzz

from finbot import rag_chain_agent as rc
from finbot.langsmith_eval import FinbotEval

langsmith_eval = FinbotEval()
collection_name = "all_docs_kb_clean_inv"
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
            {"question": "Describe the historical performance of the fund with the worst performance in the Total Return peer group."},
            []
        ),
        (
            {"question": "Describe the asset allocation of the fund with the worst performance that invests in Asia."},
            []
        ),
        (
            {"question": "Describe the historical performance of the fund with the best performance that has high UK exposure."},  # 0DFT8
            []
        ),
        (
            {"question": "Describe the asset allocation of the fund with the best performance among the funds invested in the US."},
            []
        ),
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