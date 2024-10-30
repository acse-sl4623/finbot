import os
import logging
from typing import Any, Dict
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from transformers import pipeline
from qdrant_client import models
import numpy as np
from scipy.stats import percentileofscore
from langchain import hub
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from finbot.vectorstore import QdrantVectorStore


# Custom packages
import finbot.vectorstore as vs  # noqa: F401
import finbot.prompts as prompts  # noqa: F401
from finbot.langsmith_eval import FinbotEval

langsmith_eval = FinbotEval()
prompt_react = hub.pull("hwchase17/react")
load_dotenv()

class RAGChainRoute:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.vector_store = vs.QdrantVectorStore(self.collection_name)
        self.loaded_vectorstore = self.vector_store.load_vectorstore(self.collection_name)
        self.llm_api_key = os.getenv('LLM_API_KEY')
        self.llm = Ollama(
            model="llama3:70b-instruct-q8_0",
            base_url="https://ese-timewarp.ese.ic.ac.uk",
            headers={"X-API-Key": self.llm_api_key}
        )
        #self.llm = Ollama(model="llama3", temperature=0.0)

        # Initialize logging
        logging.basicConfig(level=logging.INFO)

        # Call the method to create chains during initialization
        self.create_chains()

    def gen_build_context(self, relevant_documents):
        """
        Generates the build context by concatenating the page content of relevant documents.
        Args:
            relevant_documents (list): A list of relevant documents.
        Returns:
            str: The concatenated page content of relevant documents.
        """
        #print("Relevant documents: ", relevant_documents)
        #print("Number of relevant documents:", len(relevant_documents))
        context = ""
        for doc in relevant_documents:
            context += doc.page_content + "\n"
        #print("Context:", context)
        return context
    
    def build_exp_context(self, relevant_documents, subject: str):
        """
        Builds the exposure context based on the relevant documents and subject.
        Args:
            relevant_documents (list): A list of relevant documents.
            subject (str): The subject of the query.
        Returns:
            str: The exposure context.
        Raises:
            None
        """
        #print("Nb of relevant documents:", len(relevant_documents))
        context = ""
        for doc in relevant_documents:
            # Check if subject of query is region
            if 'region' == subject.lower():
                if 'Region Allocation' in doc.metadata.keys():
                    # context += f"{doc.page_content}\n"
                    context += f"Fund:\n"
                    context += f"  Name: {doc.metadata['fund_name']}\n"
                    context += f"  Region Allocation (%):\n"
                    for region, percentage in doc.metadata['Region Allocation'].items():
                        context += f"    {region}: {percentage}%\n"
                else:
                    context += f"The fund {doc.metadata['fund_name']} has no {subject.lower()} provided in the investment fund factsheet"
            
            # Check if subject of query is asset
            elif 'asset' == subject.lower():
                if 'Asset Allocation' in doc.metadata.keys():
                    context += f"Fund:\n"
                    context += f"  Name: {doc.metadata['fund_name']}\n"
                    context += f"  Asset Allocation (%):\n"
                    for asset_class, percentage in doc.metadata['Asset Allocation'].items():
                        context += f"    {asset_class}: {percentage}%\n"
                else:
                    context += f"The fund {doc.metadata['fund_name']} has no {subject.lower()} provided in the investment fund factsheet"

            # Check if subject of query is sector         
            elif 'sector' == subject.lower():
                if 'Sector Allocation' in doc.metadata.keys():
                    context += f"Fund:\n"
                    context += f"  Name: {doc.metadata['fund_name']}\n"
                    for sector, percentage in doc.metadata['Sector Allocation'].items():
                        context += f"    {sector}: {percentage}%\n"
                else:
                    context += f"The fund {doc.metadata['fund_name']} has no {subject.lower()} provided in the investment fund factsheet"

            # Allow for queries of multiple allocations at once
            else:
                context += doc.page_content + "\n"
        
        print("Finsihed Building Exposure Context")
        return context
    
    def build_perf_context(self, relevant_documents, subject: str):
        """
        Builds a performance context based on the relevant documents and subject.
        Args:
            relevant_documents (list): A list of relevant documents.
            subject (str): The subject of the performance context.
        Returns:
            str: The performance context.
        Raises:
            None
        """
        #print("Nb of relevant documents:", len(relevant_documents))
        context = ""
        for doc in relevant_documents:
            if 'performance' in subject.lower():
                # context += f"{doc.page_content}\n"
                context += f"Fund:\n"
                context += f"  Name: {doc.metadata['fund_name']}\n"
                # Access the "Discrete Performance" section
                if "Discrete Performance" in doc.metadata.keys():
                    context += "Discrete Performance:\n"
                    for period, performance in doc.metadata["Discrete Performance"].items():
                        context += f"  {period}:\n"
                        for key, value in performance.items():
                            context += f"    {key}: {value}\n"
                else:
                    context += f"The fund {doc.metadata['fund_name']} has no discrete performance available"
                
                if "Cumulative Performance" in doc.metadata.keys():
                    context += "Cumulative Performance:\n"
                    for period, performance in doc.metadata["Cumulative Performance"].items():
                        context += f"  {period}:\n"
                        for key, value in performance.items():
                            context += f"    {key}: {value}\n"
                else:
                    context += f"The fund {doc.metadata['fund_name']} has no cumulative performance available"

            else:
                context += doc.page_content + "\n"
        #print("Performance Context:", context)
        return context
    
    def exp_descriptive_process_documents(self, relevant_documents, subject):
        """
        Extracts relevant information from a list of documents and returns a list of dictionaries.
        Parameters:
        - relevant_documents (list): A list of documents from which information will be extracted.
        - subject (str): The subject of the documents.
        Returns:
        - list: A list of dictionaries containing the extracted information from the documents. Each dictionary has the following keys:
            - "fund_name" (str): The name of the fund extracted from the document metadata.
            - "section" (str): The section extracted from the document metadata.
            - "inv_obj" (str): The investment objective extracted from the document metadata.
            - "allocation_table" (list): The allocation table generated using the build_exp_context method.
        """
        return [
            {
                "fund_name": doc.metadata["fund_name"],  # Extract fund_name from document metadata
                "section": subject,                 # Extract section from document metadata
                "inv_obj": doc.metadata["inv_obj"],       # Extract investment objective from document metadata
                "allocation_table": self.build_exp_context([doc], subject) # Provide to prompt the allocation_table
            }
            for doc in relevant_documents
        ]

    def perf_peer_process(self, relevant_documents, given_fund):
        """
        Calculate the active performance and rank of a given fund within its peer group.
        Parameters:
        - relevant_documents (list): List of relevant documents containing performance data for funds.
        - given_fund (str): Name of the fund for which the active performance and rank are calculated.
        Returns:
        - result (dict): Dictionary containing the following information:
            - "fund_name" (str): Name of the given fund.
            - "peers" (list): List of fund names in the peer group.
            - "average_1yr_return" (float): Average 1-year return of the peer group.
            - "score_universe_rank" (float): Percentile rank of the given fund's performance within the peer group.
        Raises:
        - ValueError: If the given fund is not retrieved from the knowledge base.
        """
        # Calculate the active performance and collect all 1-year returns
        # This whole process assumes that the first retrived funds is the one we are interested in
        peer_performances = [
            doc.metadata["Cumulative Performance"]["1yr"]["Fund"]
            for doc in relevant_documents if doc.metadata["fund_name"] != given_fund
        ]

        given_fund_performance = None
        given_benchmark_performance = None

        # Find the fund and benchmark performance for the document with the given fund name
        for doc in relevant_documents:
            if doc.metadata["fund_name"] == given_fund:
                given_fund_performance = doc.metadata["Cumulative Performance"]["1yr"]["Fund"]
                given_benchmark_performance = doc.metadata["Cumulative Performance"]["1yr"]["Benchmark"]
                break  # Exit the loop once the document is found
        
        # Fund is not part of peer group
        if given_fund_performance is None:
            raise ValueError(f"Fund {given_fund} not retrieved from knowledge base")

        if given_benchmark_performance is None or given_benchmark_performance == "N/A":
            average_1yr_return= np.mean(peer_performances + given_fund_performance)  # Calculate the average 1-year return
            # Calculate the percentile rank of the first fund's performance relative to the rest
            given_fund_rank = percentileofscore(peer_performances, given_fund_performance)
        else:
            given_active_performance = given_fund_performance - given_benchmark_performance
            peer_benchmark_performances = [
                doc.metadata["Cumulative Performance"]["1yr"]["Benchmark"]
                for doc in relevant_documents if doc.metadata["fund_name"] != given_fund
            ]

            peer_active_performances = [
                perf - bench
                for perf, bench in zip(peer_performances, peer_benchmark_performances)
            ]
            average_1yr_return = np.mean(peer_active_performances + given_active_performance)
            given_fund_rank = percentileofscore(peer_active_performances, given_active_performance)

        # extracted list of peers
        peers = [doc.metadata["fund_name"] for doc in relevant_documents if doc.metadata["fund_name"] != given_fund]

        # Construct the output
        result =   {
                "fund_name": given_fund,  # Extract fund_name from document metadata
                "peers": peers,  # peer group based on investment objective similarity score
                "average_1yr_return": average_1yr_return,  # Calculate active performance
                "score_universe_rank": given_fund_rank
            }

        return result

    def exp_peer_process(self, relevant_documents, subject, given_fund):
        """
        Calculate the exposure of a given fund relative to its peers.
        Parameters:
        - relevant_documents (list): A list of relevant documents containing fund information.
        - subject (str): The subject of interest for exposure calculation.
        - given_fund (str): The name of the fund for which exposure is calculated.
        Returns:
        - result (dict): A dictionary containing the following information:
            - "fund_name" (str): The name of the given fund.
            - "peers" (list): A list of names of the fund's peers.
            - "top_exposure_name" (str): The name of the top exposure for the given fund.
            - "top_exposure_percentage" (float): The percentage of the top exposure for the given fund.
            - "section" (str): The section name for exposure calculation.
            - "average_exposure" (float): The average exposure across all funds.
            - "percentile_rank" (float): The percentile rank of the given fund's exposure relative to its peers.
        """

        # Collect all exposures for the given subject
        given_exposure_table = None

        subject = subject.capitalize()
        section_name = f"{subject} Allocation"

        # Find the fund and benchmark performance for the document with the given fund name
        for doc in relevant_documents:
            if doc.metadata["fund_name"] == given_fund:
                given_exposure_table = doc.metadata[section_name] # store the exposure dictionary
        
        
        top_exposure_name = None
        top_exposure_percentage = float('-inf')

        # Iterate through the dictionary
        for key, value in given_exposure_table.items():
            # Convert the string value to a float
            float_value = float(value)
            # Check if the current value is greater than the maximum found so far
            if float_value > top_exposure_percentage:
                top_exposure_percentage = float_value
                top_exposure_name = key

        # extracted list of peers
        peers = [doc.metadata["fund_name"] for doc in relevant_documents if doc.metadata["fund_name"] != given_fund]

        # Extract the percentages for the given fund's peers top section exposure
        peer_top_exposure_percentages = [
            float(doc.metadata[section_name][top_exposure_name])
            for doc in relevant_documents if doc.metadata["fund_name"] != given_fund
        ]
        # Calculate the percentile rank of the first fund's exposure relative to the rest
        score_fund_exposure_rank = percentileofscore(peer_top_exposure_percentages, top_exposure_percentage)

        # Extract the percentages for all the fund's top section exposure
        top_exposure_percentages = [
            float(doc.metadata[section_name][top_exposure_name])
            for doc in relevant_documents
        ]
        # Calculate the percentile rank of the first fund's exposure relative to the rest
        avg_exposure = np.mean(top_exposure_percentages)

        # Construct the output
        result = {
                "fund_name": given_fund,
                "peers": peers,
                "top_exposure_name": top_exposure_name,
                "top_exposure_percentage": top_exposure_percentage,
                "section": subject,
                "average_exposure": avg_exposure,
                "percentile_rank": score_fund_exposure_rank
        }
        # Include the average exposure
        return result
    
    def extract_fund_names(self, query):
        """
        Extracts fund names from the given query.
        Parameters:
        - query (str): The query string to extract fund names from.
        Returns:
        - list: A list of fund names extracted from the query. If no fund names are found, an empty list is returned.
        """
        # Implement a method to extract fund_name from the query
        # For example, use regex or predefined keywords
        # This is a placeholder for actual implementation
        metadata = langsmith_eval.inspect_collection_upload("all_docs_kb_clean_inv")
        all_fund_names = metadata["fund_name"]  # Use your metadata list here
        queried_fund_names = []
        for name in all_fund_names:
            if name.lower() in query.lower():
                queried_fund_names.append(name)
        if len(queried_fund_names) > 0:
            return queried_fund_names
        else:
            return []
    
    def define_simple_retriever(self, top_k=3):
        """
        Defines a simple retriever for the given top_k value.
        Parameters:
        - top_k (int): The number of points to retrieve. Default is 3.
        Returns:
        - retriever: The defined retriever object.
        """
        #print("Nb of points to retrieve:", top_k)
        retriever_kwargs = {"k": top_k}
        retriever = self.loaded_vectorstore.as_retriever(search_kwargs=retriever_kwargs)
        return retriever

    def define_score_retriever(self):
        """
        Define and return a score retriever based on the loaded vector store.
        Returns:
            retriever: A score retriever object.
        """
        retriever = self.loaded_vectorstore.as_retriever(score_threshold=0.7)
        return retriever
    
    # Helper function facillitate fund name retrieval
    def list_funds_in_knowledge_base(self, question) -> int:
        collection_name = "all_docs_kb_clean_inv"
        vs = QdrantVectorStore(collection_name)
        scroll_response = vs.client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_vectors=True,
            with_payload=True
        )
        points = scroll_response[0]
        # Count the number of documents in the collection
        names = []
        for point in points:
            names.append(point.payload["metadata"]["fund_name"])
        return names

    def define_filtered_full_doc_retriever(self, query, top_k = 1):
        """
        Defines a filtered full document retriever based on the given query and top_k value.
        Parameters:
        - query (str): The query string used to extract fund names.
        - top_k (int): The number of top results to retrieve. Default is 1.
        Returns:
        - retriever: The filtered full document retriever object.
        Raises:
        - None
        Example usage:
        retriever = define_filtered_full_doc_retriever("query string", top_k=5)
        """
        search_names_str = self.extract_fund_names_chain.invoke(query)["text"] # Extract fund names from the query
        search_names = search_names_str.split(",")
        all_fund_names = self.list_funds_in_knowledge_base(query)
        filter_condition = models.Filter(must=[], should=[])
        for search_name in search_names:
            # Identify best match for the search_name in the list of all fund names
            best_match = process.extractOne(search_name, all_fund_names, scorer=fuzz.ratio)
            fund_name = best_match[0]

            # Pass the fund name to the filter condition
            filter_condition.should.append(
                models.FieldCondition(
                    key="metadata.fund_name",
                    match=models.MatchValue(value=fund_name)
                )
            )
        
        # Check if filter_condition is empty by inspecting its must and should lists
        if filter_condition.must or filter_condition.should:
            retriever_kwargs = {"filter": filter_condition}
        else:
            retriever_kwargs = {"k": top_k}
        
        # We apply filtered retrieval for the full document from the docs knowledge base
        collection_name = "all_docs_kb_clean_inv"
        vector_store = vs.QdrantVectorStore(collection_name)
        loaded_vectorstore = vector_store.load_vectorstore(collection_name)
        retriever = loaded_vectorstore.as_retriever(search_kwargs=retriever_kwargs)
        return retriever
    
    def get_simple_retriever(self, top_k):
        return self.define_simple_retriever(top_k)
    
    def get_score_retriever(self):
        return self.define_score_retriever()
    
    def get_filtered_fund_name_retriever(self, query, top_k):
        return self.define_filtered_full_doc_retriever(query, top_k)
    
    def get_summarized_text(self, text: str) -> str:
        summarizer = pipeline("summarization", model="Falconsai/text_summarization")
        return summarizer(text, max_length=1000, min_length=30, do_sample=False)[0]['summary_text']

    def simple_task_first_route(self, info):
        print("Incoming info (first route):", info)
        response = info["task"]["text"]
        print("Task: ", response)
        if "specific" in response.lower():
            return self.general_chain
        else:
            # Include top_k in the data passed to the next chain
            return self.simple_second_chain #, {"top_k": info['task']["top_k"]}
        
    def guided_task_first_route(self, info):
        print("Incoming info (first route):", info)
        response = info["task"]["text"]
        print("Task: ", response)
        if "specific" in response.lower():
            return self.general_chain
        else:
            return self.guided_second_chain #, {"top_k": info["top_k"]}

    def guided_second_route(self, info):
        print("Incoming info (second route):", info)
        subject = info["subject"]["text"]
        task = info["task"]["text"]
        print("Subject: ", subject)

        # Task: Describe
        if "describe" in task.lower():
            if "performance" in subject.lower():
                return self.guided_perf_chain
            elif "sector" in subject.lower() or "region" in subject.lower() or "asset" in subject.lower():
                return self.guided_exp_chain
            else:
                return self.simple_description_chain
        
        # Task: Compare
        elif "compare" in task.lower():
            if "performance" in subject.lower():
                return self.comp_perf_chain
            elif "sector" in subject.lower() or "region" in subject.lower() or "asset" in subject.lower():
                return self.comp_exp_chain
            else:
                return self.general_chain
        
        # Task: Analyze
        elif "analyze" in task.lower():
            if "performance" in subject.lower():
                return self.analyze_peer_perf_score
            elif "sector" in subject.lower() or "region" in subject.lower() or "asset" in subject.lower():
                return self.analyze_peer_exp_score
            else:
                return self.general_chain

    def simple_second_route(self, info):
        """
        This method determines the appropriate chain based on the subject of the given info.
        Parameters:
        - info (dict): A dictionary containing information about the subject.
        Returns:
        - chain (list): The chain that corresponds to the subject of the info.
        """
        #print("Incoming info (second route):", info)
        subject = info["subject"]["text"]
        print("Subject: ", subject)
        if "performance" in subject.lower():
            return self.simple_description_chain
        elif "sector" in subject.lower() or "region" in subject.lower() or "asset" in subject.lower():
            return self.simple_description_chain
        else:
            return self.general_chain
    
    def create_chains(self):
        """
        Create chains for different functionalities.
        This method initializes and assigns various chains for different functionalities in the class.
        Each chain is responsible for a specific task or operation.
        Returns:
            None
        """
        # Define all the chains here
        self.task_classification_chain = (
            prompts.task_classification_template 
            | self.llm 
            | RunnableLambda(lambda output: {"text": output})
        )

        self.subject_classification_chain = (
            prompts.subject_classification_template 
            | self.llm 
            | RunnableLambda(lambda output: {"text": output})
        )

        self.extract_fund_names_chain = (
            prompts.extract_fund_names_template 
            | self.llm 
            | RunnableLambda(lambda output: {"text": output})
        )

        # Second chain routing to the simple prompts
        self.simple_second_chain = ({
            "subject": self.subject_classification_chain, # assign the returned value to "subject" key

            "question": lambda x: x["question"], # pass on the "question" key from first chain
            "top_k": lambda x: x["top_k"], # pass on the "top_k" key from first chain
            "task": lambda x: x["task"] # pass on the "task" key from first chain
        } | RunnableLambda(self.simple_second_route)
        )

        # Second chain routing to the guided prompts
        self.guided_second_chain = ({
            "subject": self.subject_classification_chain, # assign the returned value to "subject" key

            "question": lambda x: x["question"], # pass on the "question" key from first chain
            "top_k": lambda x: x["top_k"], # pass on the "top_k" key from first chain
            "task": lambda x: x["task"] # pass on the "task" key from first chain
        } | RunnableLambda(self.guided_second_route)
        )

        # General chain (The general chain uses simple retrieval to show routing efficiency of Agent)
        self.general_chain = (
            {
            "context": RunnableLambda(lambda inputs: 
                self.gen_build_context(
                    self.get_simple_retriever(inputs["top_k"]).invoke(inputs["question"]))),
            "question": RunnablePassthrough()                                                                     
            }
            | prompts.general_template
            | self.llm
            | StrOutputParser()
        )

        # Simple descriptive template
        self.simple_description_chain = (
            {
                "context": RunnableLambda(lambda inputs: 
                    self.gen_build_context(
                        self.get_simple_retriever(inputs["top_k"]).invoke(inputs["question"]))),
                "question": RunnablePassthrough()
            }
            | prompts.general_template
            | self.llm
            | StrOutputParser()
        )

        # Guided CoT performance template
        self.guided_perf_chain = (
            {
            "context": RunnableLambda(lambda inputs: 
                self.build_perf_context(
                    self.get_filtered_fund_name_retriever(inputs["question"], top_k = 1).invoke(inputs["question"]),
                    inputs["subject"]["text"]
                )
            ),
            "question": RunnablePassthrough()
            }
            | prompts.guided_description_perf_template
            | self.llm
            | StrOutputParser()
        )

        # Guided CoT exposure template
        # We only pass the exposure and investment objective to context:
        self.guided_exp_chain = (
            {
                "context": RunnableLambda(lambda inputs: 
                    self.exp_descriptive_process_documents(
                        self.get_filtered_fund_name_retriever(inputs["question"], top_k = 1).invoke(inputs["question"]),
                        inputs["subject"]["text"]
                    )
                )
            } # all the input variables are returned by exp_descriptive_process_documents under key "context"
            | RunnableLambda(lambda inputs: {
                "fund_name": inputs["context"][0]["fund_name"], 
                "inv_obj": inputs["context"][0]["inv_obj"], 
                "section": inputs["context"][0]["section"],
                "allocation_table": inputs["context"][0]["allocation_table"]
            })
            | prompts.guided_description_exp_template
            | self.llm
            | StrOutputParser()
        )

        # Comparison of exposure
        self.comp_exp_chain = (
            {
                "context": RunnableLambda(lambda inputs: 
                    self.build_exp_context(
                        self.get_filtered_fund_name_retriever(inputs["question"], top_k = 2).invoke(inputs["question"]),
                        inputs["subject"]["text"]
                    )
                ),
            "question": RunnablePassthrough()
        }
            | prompts.compare_funds_exp_template
            | self.llm
            | StrOutputParser()
        )

        # Comparison of exposure
        self.comp_perf_chain = (
            {
                "context": RunnableLambda(lambda inputs: 
                    self.build_perf_context(
                        self.get_filtered_fund_name_retriever(inputs["question"], top_k = 2).invoke(inputs["question"]),
                        inputs["subject"]["text"]
                    )
                ),
            "question": RunnablePassthrough()
        }
            | prompts.compare_funds_perf_template
            | self.llm
            | StrOutputParser()
        )

        # Pass into this chain as well the output from the descriptive performance chain
        self.analyze_peer_perf_score = (
            # Step 2a: Extract fund_name from the input question
            self.extract_fund_names_chain
            | RunnableLambda(lambda inputs: {
                # Step 2b: Use the extracted fund_name to create an optimized query
                "question": f"What funds have investment objectives similar to {inputs}?"
            })
            | {
                # Step 2c: Process the documents to find the most similar investment objectives
                "context": RunnableLambda(lambda inputs:
                    self.perf_peer_process(
                        self.get_score_retriever().invoke(inputs["question"]),
                        inputs["fund_names"]
                    )
                )
            }
            | RunnableLambda(lambda inputs: {
                "fund_name": inputs["context"]["fund_name"],
                "peers": inputs["context"]["peers"],
                #"inv_objs": inputs["context"]["inv_objs"],
                "average_1yr_return": inputs["context"]["average_1yr_return"],
                "score_universe_rank": inputs["context"]["score_universe_rank"]
            })
            | prompts.analyze_peer_perf_score_template  # Step 2e: Formulate the query to find similar funds
            | self.llm  # Step 2f: Pass the query to the LLM to get the similar funds
            | StrOutputParser()  # Step 2g: Parse the output to get the final results
        )

        # Pass into this chain as well the output from the descriptive exposure chain
        self.analyze_peer_exp_score = (
            # Step 2a: Extract fund_name from the input question
            self.extract_fund_names_chain
            | RunnableLambda(lambda inputs: {
                # Step 2b: Use the extracted fund_name to create an optimized query
                "question": f"What funds have investment objectives similar to {inputs['fund_names']}?"
            })
            | {
                # Step 2c: Process the documents to find the most similar investment objectives
                "context": RunnableLambda(lambda inputs:
                    self.exp_peer_process(
                        self.get_score_retriever().invoke(inputs["question"]),
                        inputs["fund_names"],
                        inputs["subject"]["text"]
                    )
                )
            }
            | RunnableLambda(lambda inputs: {
                "fund_name": inputs["context"]["fund_name"],
                "peers": inputs["context"]["peers"],
                "top_exposure_name": inputs["context"]["top_exposure_name"],
                "top_exposure_percentage": inputs["context"]["top_exposure_percentage"],
                "section": inputs["context"]["section"],
                "average_exposure": inputs["context"]["average_exposure"],
                "percentile_rank": inputs["context"]["percentile_rank"]
            })
            | prompts.analyze_peer_exp_score_template  # Step 2e: Formulate the query to find similar funds
            | self.llm  # Step 2f: Pass the query to the LLM to get the similar funds
            | StrOutputParser()  # Step 2g: Parse the output to get the final results
        )
     
    def execute_simple_full_chain(self, query: str, top_k=1) -> Any:
        """
        Executes a simple full chain for the given query.
        Args:
            query (str): The query to be executed.
            top_k (int, optional): The number of results to return. Defaults to 1.
        Returns:
            Any: The result of the execution.
        """
        # Define the input for the chain, including top_k
        input_data = {
            "question": query,
            "top_k": top_k
        }

        # Create the first chain using task classification
        first_chain = {
            "task": self.task_classification_chain, 
            "question": lambda x: x["question"],
            "top_k": lambda x: x["top_k"]
        } | RunnableLambda(self.simple_task_first_route)

        # Invoke the first chain and return the result
        return first_chain.invoke(input_data)

    def execute_guided_full_chain(self, query: str, top_k=1) -> Any:
        """
        Executes the guided full chain for a given query.
        Args:
            query (str): The query to be processed.
            top_k (int, optional): The number of top results to retrieve. Defaults to 1.
        Returns:
            Any: The result of the full chain execution.
        """

        # Define the input for the chain
        input_data = {
            "question": query,
            "top_k": top_k
        }

        # Create the first chain using task classification
        first_chain = {
            "task": self.task_classification_chain,
            "question": lambda x: x["question"],
            "top_k": lambda x: x["top_k"]
        } | RunnableLambda(self.guided_task_first_route)

        # Invoke the full chain and return the result
        return first_chain.invoke(input_data)
