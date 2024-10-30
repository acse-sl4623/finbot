import sys
import os
import asyncio
import time
import random

# Add the root directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from langsmith import Client
from langsmith.utils import LangSmithError
from langchain.smith.evaluation.config import RunEvalConfig
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from langchain.smith.evaluation.runner_utils import run_on_dataset
from langchain_community.llms import Ollama
from langsmith.evaluation import LangChainStringEvaluator, evaluate, StringEvaluator
from langsmith.evaluation import evaluate_comparative
from datetime import datetime

from finbot.vectorstore import QdrantVectorStore

class FinbotEval:
    """
    Class representing the evaluation of Finbot.
    Methods:
    - __init__: Initializes the FinbotEval object and sets up the necessary credentials and clients.
    - create_dataset: Creates a new dataset with the given name, description, and example queries.
    - run_evaluation_on_dataset: Runs the evaluation on the specified dataset using the provided chain execution function.
    - inspect_collection_upload: Inspects the uploaded collection and provides information about the vectors and metadata.
    Attributes:
    - LANGSMITH_API_KEY: The API key for LangSmith.
    - LLM_API_KEY: The API key for LLM.
    - ls_client: The LangSmith client.
    - llm: The LLM client.
    - eval_llm: The LLM client for evaluation.
    """
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize LangSmith credentials
        self.LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
        os.environ["LANGCHAIN_TRACING"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "Finbot"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = self.LANGSMITH_API_KEY

        # Initialize LangSmith Client
        self.ls_client = Client()

        # LLM Initialization
        self.LLM_API_KEY = os.getenv('LLM_API_KEY')
        self.llm = Ollama(
            model="llama3:70b-instruct-q8_0",
            base_url="https://ese-timewarp.ese.ic.ac.uk",
            headers={"X-API-Key": self.LLM_API_KEY}
        )

        self.eval_llm = Ollama(
            model="llama3:8b-instruct-q8_0",
            base_url="https://ese-timewarp.ese.ic.ac.uk",
            headers={"X-API-Key": self.LLM_API_KEY}
        )

        # Explored but ultimately not implemented as evaluation criteria due to time complexity
        self.qa_evaluator = LangChainStringEvaluator(
                "cot_qa",
                config={
                    "llm": self.eval_llm,
                },
                prepare_data=lambda run, example: {
                    "prediction": run.outputs["output"],
                    "reference": example.outputs["context"],
                    "input": example.inputs["question"],
                },
            )
        
        # Explored but ultimately not implemented as evaluation criteria due to time complexity
        self.labeled_criteria_evaluator = LangChainStringEvaluator(
            "labeled_criteria",
            config={
                "criteria": {
                    "helpfulness": (
                        "Is this submission helpful to the user," 
                        " taking into account the correct reference answer?"
                    )
                },
                "llm": self.eval_llm,
            },
            prepare_data=lambda run, example: {
                "prediction": run.outputs["output"],
                "reference": example.outputs["context"],
                "input": example.inputs["question"],   
            }
        )

        # Explored but ultimately not implemented as evaluation criteria due to time complexity
        self.labeled_score_evaluator = LangChainStringEvaluator(
        "labeled_criteria",
        config={
            "criteria": {
                "accuracy": """Is the Assistant's Answer grounded in the Ground Truth documentation? A score of [[1]] means that the
                Assistant answer contains is not at all based upon / grounded in the Ground Truth documentation. A score of [[5]] means 
                that the Assistant answer contains some information (e.g., a hallucination) that is not captured in the Ground Truth 
                documentation. A score of [[10]] means that the Assistant answer is fully based upon the in the Ground Truth documentation."""
            },
            "normalize_by": 10,
            "llm": self.eval_llm,
        },
        prepare_data=lambda run, example: {
            "prediction": run.outputs["output"],
            "reference": example.outputs["context"],
            "input": example.inputs["question"],
        }
    )
        
    def create_dataset(self, dataset_name, dataset_description, example_queries):
        """
        Creates a dataset with the given name and description, and adds example queries to it.
        Args:
            dataset_name (str): The name of the dataset.
            dataset_description (str): The description of the dataset.
            example_queries (list): A list of example queries to be added to the dataset.
        Returns:
            dataset: The created dataset.
        Raises:
            LangSmithError: If an error occurs during dataset creation.
        """
        try:
            # Check if the dataset already exists
            datasets = self.ls_client.list_datasets()
            existing_dataset = next((ds for ds in datasets if ds.name == dataset_name), None)

            if existing_dataset:
                # Load existing dataset
                dataset = existing_dataset
                print(f"Dataset '{dataset_name}' already exists. Loaded existing dataset.")
            else:
                # Create new dataset
                dataset = self.ls_client.create_dataset(
                    dataset_name=dataset_name,
                    description=dataset_description
                )
                print(f"Dataset '{dataset_name}' created successfully.")

        except LangSmithError as e:
            print(f"An error occurred: {e}")

        for input_query in example_queries:
            self.ls_client.create_example(
                inputs={"question": input_query},
                outputs=None,
                dataset_id=dataset.id
            )

        return dataset

    def run_evaluation_on_dataset(self, dataset_name, chain_execution_function, run_name, top_k = 4):
        """
        Run evaluation on a dataset using the specified chain execution function.
        Args:
            dataset_name (str): The name of the dataset to run evaluation on.
            chain_execution_function (function): The function that executes the chain.
            run_name (str): The name of the run.
            top_k (int, optional): The number of top results to consider. Defaults to 4.
        Raises:
            TypeError: If a type error occurs during execution.
            Exception: If an unexpected error occurs during execution.
        """
        # Create a session with retry and increased timeout
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('https://', adapter)
        print("Processing chain: ", chain_execution_function.__name__)
        # Set timeout
        timeout = 10000
        ls_client = Client(session=session, timeout_ms=timeout)

        eval_config = RunEvalConfig(
            evaluators=[
                "cot_qa", # correctness relative to the reference answer
                RunEvalConfig.Criteria("coherence"),
                ],
            eval_llm=self.eval_llm,
            verbose=True
        )

        try:
            current_time = datetime.now()
            # Format the datetime as you like
            formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
            project_name = f"{run_name} - {formatted_time}"
            print("Running the 'run_on_dataset' method...")

            # Evaluation of Agent Chain
            if "agent" in chain_execution_function.__name__.lower():
                run_on_dataset(
                    client=ls_client,
                    dataset_name=dataset_name,
                    project_name = project_name,
                    llm_or_chain_factory=lambda inputs: chain_execution_function(inputs)["output"],
                    evaluation=eval_config,
                )

            # Evaluation of Route Chain
            elif "guided" in chain_execution_function.__name__.lower():
                run_on_dataset(
                    client=ls_client,
                    dataset_name=dataset_name,
                    project_name = project_name,
                    llm_or_chain_factory=lambda inputs: chain_execution_function(inputs["question"]),
                    evaluation=eval_config,
                )

            # Evaluation of all other chains
            else:
                run_on_dataset(
                    client=ls_client,
                    dataset_name=dataset_name,
                    project_name = project_name,
                    llm_or_chain_factory=chain_execution_function,
                    evaluation=eval_config,
                )

        except TypeError as te:
            print(f"TypeError occurred: {te}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def inspect_collection_upload(self, collection_name, point_verbose=False):
        """
        Inspects the collection upload and returns metadata information.
        Args:
            collection_name (str): The name of the collection to inspect.
            point_verbose (bool, optional): Whether to print verbose information about each point. Defaults to False.
        Returns:
            dict: A dictionary containing the extracted metadata information.
        
        This function also returns the unique metadata information extracted from the collection.
        """
        vs = QdrantVectorStore(collection_name)

        scroll_response = vs.client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_vectors=True,
            with_payload=True
        )

        points = scroll_response[0]
        print(f"Number of points in collection {collection_name}: {len(points)}", '\n')

        vectors = []
        vector_length = None
        consistent_shape = True

        # Extract unique metadata values
        asset_classes = []
        fund_names = []
        launch_dates = []
        isins = []
        sedols = []
        inv_objs = []

        for point in points:
            if point_verbose:
                print("Payload", point.payload)
                print("Vector", point.vector, '\n')

            # Extract metadata sets from payload
            metadata = point.payload['metadata']
            if metadata['asset_class'] not in asset_classes:
                asset_classes.append(metadata['asset_class'])
            if metadata['fund_name'] not in fund_names:
                fund_names.append(metadata['fund_name'])
            if metadata['launch_date'] not in launch_dates:
                launch_dates.append(metadata['launch_date'])
            if metadata['ISIN'] not in isins:
                isins.append(metadata['ISIN'])
            if metadata['SEDOL'] not in sedols:
                sedols.append(metadata['SEDOL'])
            if metadata['inv_obj'] not in inv_objs:
                inv_objs.append(metadata['inv_obj'])

            if vector_length is None:
                vector_length = len(point.vector)
            elif len(point.vector) != vector_length:
                consistent_shape = False
            vectors.append(tuple(point.vector))  # Convert list to tuple for hashability

        # Check vector consistency
        if consistent_shape:
            print(f"All vectors have the same size and shape: {vector_length}")
        else:
            print("Not all vectors have the same size and shape.")

        # Check vector uniqueness
        unique_vectors = set(vectors)
        if len(vectors) == len(unique_vectors):
            print("All vector embeddings are unique.")
        else:
            print("There are duplicate vector embeddings.")

        metadata = {
            "asset_class": asset_classes,
            "fund_name": fund_names,
            "launch_date": launch_dates,
            "ISIN": isins,
            "SEDOL": sedols,
            "inv_obj": inv_objs
        }
        return metadata