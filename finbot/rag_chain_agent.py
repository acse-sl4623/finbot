import os
import logging
from typing import Any, Dict
import numpy as np
from scipy.stats import percentileofscore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from transformers import pipeline
from qdrant_client import models
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from langchain.agents import Tool
from langchain.prompts import PromptTemplate

import finbot.vectorstore as vs  # noqa: F401
import finbot.prompts as prompts  # noqa: F401
from finbot.vectorstore import QdrantVectorStore
from finbot.langsmith_eval import FinbotEval

langsmith_eval = FinbotEval()
prompt_react = hub.pull("hwchase17/react")
load_dotenv()

class RAGChain:
    """
    RAGChain class represents a chain of functions for generating context and performing various operations on financial documents.
    Methods:
    - gen_build_context(relevant_documents): Generates the build context by concatenating the page content of relevant documents.
    - gen_build_list_context(output): Generates the context for building a list.
    - build_exp_context(relevant_documents, subject): Builds the exposure context based on the relevant documents and subject.
    - exp_descriptive_process_documents(relevant_documents, subject): Process a list of relevant documents and extract relevant information for descriptive analysis.
    - build_perf_context(relevant_documents, subject): Builds the performance context based on the relevant documents and subject.
    - compare_peer_performance_context(relevant_documents, given_fund): Compare the performance of a given fund with its peers based on relevant documents.
    - compare_peer_exp_context(relevant_documents, given_fund): Compare the exposure context of a given fund with its peers.
    - list_funds_in_knowledge_base(question): Helper function to facilitate fund name retrieval.
    - define_filtered_full_doc_retriever(query, top_k): Defines a filtered full document retriever based on the given query and top_k value.
    - define_score_retriever(): Defines and returns a score retriever based on the loaded vector store.
    - define_score_retriever_with_k(top_k): Defines a score retriever with a given top_k value.
    - get_filtered_full_doc_retriever(query, top_k): Wrapping function to get the filtered full document retriever.
    - get_score_retriever(): Wrapping function to get the score retriever.
    - get_score_retriever_with_k(top_k): Wrapping function to get the score retriever with a given top_k value.
    - create_peer_group(question): Creates a peer group based on the given question.
    """
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.vector_store = vs.QdrantVectorStore(self.collection_name)
        self.loaded_vectorstore = self.vector_store.load_vectorstore(
            self.collection_name)
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

    # 1. Context 
    
    # 1.1 General Context Building Function
    def gen_build_context(self, relevant_documents):
        """
        Generates the build context by concatenating the page content of relevant documents.
        Args:
            relevant_documents (list): A list of relevant documents.
        Returns:
            str: The concatenated page content of relevant documents.
        """
        # print("Relevant documents: ", relevant_documents)
        # print("Number of relevant documents:", len(relevant_documents))
        context = ""
        for doc in relevant_documents:
            context += doc.page_content + "\n"
        # print("Context:", context)
        return context
    
    # 1.2. List Context Building Function
    def gen_build_list_context(self, output):
        """
        Generate the context for building a list.

        Parameters:
            output (list): The list of documents.

        Returns:
            str: The generated context for building a list.
        """
        context = ""#"Here is a list of all the funds in the knowledge base: " + "\n"
        for idx, doc in enumerate(output):
            context += str(idx)+ '. ' + str(doc) + "\n"
        #print("List Context:", context)
        return context
    
    # 1.3. Build Exposure Context
    def build_exp_context(self, relevant_documents, subject: dict):
        """
        Builds the exposure context based on the relevant documents and subject.
        Args:
            relevant_documents (list): A list of relevant documents.
            subject (dict): The subject of the query.
        Returns:
            str: The exposure context.
        Raises:
            None
        """
        #print("Nb of relevant documents:", len(relevant_documents))
        subject = subject["text"]
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
        
        #print("Finished Building Exposure Context")
        return context
    
    # 1.4 Build Exposure Context for description
    def exp_descriptive_process_documents(self, relevant_documents, subject):
        """
        Process a list of relevant documents and extract relevant information for descriptive analysis.
        Args:
            relevant_documents (list): A list of relevant documents to be processed.
            subject (str): The subject of the analysis.
        Returns:
            list: A list of dictionaries containing the extracted information from each document.
                Each dictionary has the following keys:
                - "fund_name": The name of the fund extracted from the document metadata.
                - "section": The section extracted from the document metadata.
                - "inv_obj": The investment objective extracted from the document metadata.
                - "allocation_table": The result of the `build_exp_context` method applied to the document.
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
    
    # 1.5 Build Performance Context
    def build_perf_context(self, relevant_documents, subject: dict):
        """
        Builds the performance context based on the relevant documents and subject.
        Args:
            relevant_documents (list): A list of relevant documents.
            subject (dict): The subject containing the text.
        Returns:
            str: The performance context.
        """
        #print("Nb of relevant documents:", len(relevant_documents))
        subject = subject["text"]
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
    
    # 1.6. Build Peer Performance Comparison Context
    def compare_peer_performance_context(self, relevant_documents:list, given_fund:dict):
        # Calculate the active performance and collect all 1-year returns
        print("Nb of peers:", len(relevant_documents))
        given_fund = given_fund["text"] # extract the string from the dictionary

        # Ensure we get the Document object for the queried fund even if for some reason it's not in peers
        given_fund_doc = self.get_filtered_full_doc_retriever(given_fund).invoke(given_fund)[0]

        # Extract Cumulative Performance attributes
        if "Cumulative Performance" in given_fund_doc.metadata.keys():
            given_fund_performance = float(given_fund_doc.metadata["Cumulative Performance"]["1yr"]["Fund"])
            if "Benchmark" in given_fund_doc.metadata["Cumulative Performance"]["1yr"].keys():
                if given_fund_doc.metadata["Cumulative Performance"]["1yr"]["Benchmark"] != "N/A":
                    given_benchmark_performance = float(given_fund_doc.metadata["Cumulative Performance"]["1yr"]["Benchmark"])
                else:
                    given_benchmark_performance = "N/A"
            else:
                given_benchmark_performance = "N/A"
        
        # Extract Discrete Performance attributes
        elif "Discrete Performance" in given_fund_doc.metadata.keys():
            given_fund_performance = float(given_fund_doc.metadata["Discrete Performance"]["0-12m"]["Fund"])
            if "Benchmark" in given_fund_doc.metadata["Discrete Performance"]["0-12m"].keys():
                if given_fund_doc.metadata["Discrete Performance"]["0-12m"]["Benchmark"] != "N/A":
                    given_benchmark_performance = float(given_fund_doc.metadata["Discrete Performance"]["0-12m"]["Benchmark"])
                else:
                    given_benchmark_performance = "N/A"
            else:
                given_benchmark_performance = "N/A"
        else:
            result =   {
                "fund_name": given_fund,  # Extract fund_name from document metadata
                "peers": "No peers due to fund not having performance data",  # peer group based on similarity score
                "avg_1yr_return": "N/A",  # Calculate active performance
                "score_universe_rank": "N/A"
            }
            return result

        # Extract peers
        peer_performances = []
        peer_benchmark_performances =[]
        peer_active_performances = []
        peers = []

        for i, doc in enumerate(relevant_documents):
            if doc.metadata["fund_name"] != given_fund:
                if "Cumulative Performance" in doc.metadata.keys():
                    peer_performance = float(doc.metadata["Cumulative Performance"]["1yr"]["Fund"])
                    peer_performances.append(peer_performance)
                    peers.append(doc.metadata["fund_name"])
                    #print("Peer Performance:", peer_performance)
                    if "Benchmark" in doc.metadata["Cumulative Performance"]["1yr"].keys():
                        if doc.metadata["Cumulative Performance"]["1yr"]["Benchmark"] != "N/A":
                            peer_benchmark_performance = float(doc.metadata["Cumulative Performance"]["1yr"]["Benchmark"])
                            peer_benchmark_performances.append(peer_benchmark_performance)
                            peer_active_performances.append(peer_performance - peer_benchmark_performance)
                elif "Discrete Performance" in doc.metadata.keys():
                    peer_performance = float(doc.metadata["Discrete Performance"]["0-12m"]["Fund"])
                    peer_performances.append(peer_performance)
                    peers.append(doc.metadata["fund_name"])
                    if "Benchmark" in doc.metadata["Discrete Performance"]["0-12m"].keys():
                        if doc.metadata["Discrete Performance"]["0-12m"]["Benchmark"] != "N/A":
                            peer_benchmark_performance = float(doc.metadata["Discrete Performance"]["0-12m"]["Benchmark"])
                            peer_benchmark_performances.append(peer_benchmark_performance)
                            peer_active_performances.append(peer_performance - peer_benchmark_performance)
                else:
                    continue # Skip the fund if no performance data is available
            else:
                continue # Skip if given fund identified among peers
        
        if given_benchmark_performance is None or given_benchmark_performance == "N/A":
            average_1yr_return= np.mean(peer_performances + [given_fund_performance])  # Calculate the average 1-year return
            given_fund_rank = percentileofscore(peer_performances, given_fund_performance) # Calculate the percentile rank within peer group
        else:
            given_active_performance = given_fund_performance - given_benchmark_performance
            average_1yr_return = np.mean(peer_active_performances + [given_active_performance])
            given_fund_rank = percentileofscore(peer_active_performances, given_active_performance)

        # Extracted list of peers
        print("Peers:", peers)

        # Construct the output
        result =   {
                "fund_name": given_fund,  # Extract fund_name from document metadata
                "peers": peers,  # peer group based on similarity score
                "avg_1yr_return": average_1yr_return,  # Calculate active performance
                "score_universe_rank": given_fund_rank
            }

        return result
    
    # 1.7. Build Peer Exposure Comparison Context
    def compare_peer_exp_context(self, relevant_documents:list, given_fund:dict):
        """
        Compare the exposure context of a given fund with its peers.
        Args:
            relevant_documents (list): A list of relevant documents containing metadata of funds.
            given_fund (dict): A dictionary containing the text of the given fund.
        Returns:
            dict: A dictionary containing the comparison results including the given fund's name, its peers,
                    the top exposure name, the top exposure percentage, the section, the average exposure,
                    and the percentile rank of the given fund's exposure relative to the rest.
        """

        given_fund = given_fund["text"]
        subject= self.subject_classification_chain.invoke(given_fund)["text"]
        # Collect all exposures for the given subject
        subject = subject.capitalize()
        section_name = f"{subject} Allocation"

        # Ensure we get the Document object for the queried fund even if for some reason it's not in peers
        given_fund_doc = self.get_filtered_full_doc_retriever(given_fund).invoke(given_fund)[0]
        print("Given Fund Document Metadata Keys: ", given_fund_doc.metadata.keys())
        given_exposure_table = given_fund_doc.metadata[section_name]

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

    # 2. Define Retrievers
    # 2.1. Define Filtered Retrieval Based on Fund Name
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
    
    # 2.2. Define Score Retrieval with 80% Similarity
    # Peer group is defined by 80% similarity with investment objective
    def define_score_retriever(self):
        """
        Defines and returns a score retriever based on the loaded vector store.
        Returns:
            retriever: A score retriever object with a score threshold of 0.8 (80% similarity).
        """
        retriever = self.loaded_vectorstore.as_retriever(score_threshold=0.8) # 80% similarity
        return retriever
    
    # 2.3. Define Score Retrieval with Top K
    def define_score_retriever_with_k(self, top_k):
        """
        Defines a score retriever with a given top_k value.
        Parameters:
            top_k (int): Number of top results to retrieve.
        Returns:
            retriever: The created retriever object.
        """
        retriever_kwargs = {
            "k": top_k,              # Number of top results to retrieve
            "score_threshold": 0.8  # Minimum score threshold for the results
        }
        # Create the retriever using the retriever_kwargs
        retriever = self.loaded_vectorstore.as_retriever(search_kwargs=retriever_kwargs)
        return retriever

    # 3. Define wrapping get functions for retrievers
    def get_filtered_full_doc_retriever(self, query, top_k=1):
        return self.define_filtered_full_doc_retriever(query, top_k)
    
    def get_score_retriever(self):
        return self.define_score_retriever()
    
    def get_score_retriever_with_k(self, top_k):
        return self.define_score_retriever_with_k(top_k)
    
    def create_peer_group(self, question):
        """
        Creates a peer group based on the given question.
        Parameters:
        - question (str): The question used to retrieve scores for the peers.
        Returns:
        - fund_ranks (dict): A dictionary containing the rank and peer names for each fund in the peer group.
        """
        fund_ranks = {}
        peers = self.get_score_retriever().invoke(question)
        peer_names = [doc.metadata["fund_name"] for doc in peers]
        print("Peers:", peer_names)
        if len(peers) < 1:
            return "No funds identified"
        else:
            peer_active_performances = {}
            fund_ranks = {}
            # Calculate percentile rank for each peer within the peer group
            for i, doc in enumerate(peers):
                if "Cumulative Performance" in doc.metadata.keys():
                    peer_performance = float(doc.metadata["Cumulative Performance"]["1yr"]["Fund"])
                    #print("Peer Performance:", peer_performance)
                    if "Benchmark" in doc.metadata["Cumulative Performance"]["1yr"].keys():
                        if doc.metadata["Cumulative Performance"]["1yr"]["Benchmark"] != "N/A":
                            peer_benchmark_performance = float(doc.metadata["Cumulative Performance"]["1yr"]["Benchmark"])
                            peer_active_performances[doc.metadata["fund_name"]] = peer_performance - peer_benchmark_performance
                            #print("Peer Active Performances:", peer_active_performances)
                elif "Discrete Performance" in doc.metadata.keys():
                    peer_performance = float(doc.metadata["Discrete Performance"]["0-12m"]["Fund"])
                    #print("Peer Performance:", peer_performance)
                    if "Benchmark" in doc.metadata["Discrete Performance"]["0-12m"].keys():
                        if doc.metadata["Discrete Performance"]["0-12m"]["Benchmark"] != "N/A":
                            peer_benchmark_performance = float(doc.metadata["Discrete Performance"]["0-12m"]["Benchmark"])
                            peer_active_performances[doc.metadata["fund_name"]] = peer_performance - peer_benchmark_performance
                            #print("Peer Active Performances:", peer_active_performances)
                else:
                    continue # Skip the fund if no performance data is available
            #print("Peer Active Performances:", peer_active_performances)
            for fund_name, active_performance in peer_active_performances.items():
                # Calculate percentile rank of this active performance within the peer group
                rank = percentileofscore(np.array(list(peer_active_performances.values()), dtype=float), active_performance)
                # Store the result in fund_ranks
                fund_ranks[fund_name] = {"rank": rank, "peers": peer_names}
            return fund_ranks
    
    # 4. Define Tools (Functions)
    # 4.1. Identify Top Performer
    def identify_top_performer(self, question):
        """
        Identifies the top performer fund within a given peer group based on fund ranks.
        Parameters:
        - question (str): The question related to the peer group.
        Returns:
        - str: The name of the top performer fund within the peer group.
        Raises:
        - None
        """
        fund_ranks = self.create_peer_group(question)
        if len(fund_ranks) == 0:
            return "No funds identified for this peer group"
        else:
            top_performer = max(fund_ranks, key=lambda fund_name: fund_ranks[fund_name]["rank"])
            return top_performer
    
    # 4.2. Identify Worst Performer
    def identify_worst_performer(self, question):
        """
        Identifies the worst performer among the funds in the given peer group.
        Parameters:
        - question (str): The question related to the peer group.
        Returns:
        - str: The name of the worst performing fund.
        Raises:
        - None
        """
        fund_ranks = self.create_peer_group(question)
        if len(fund_ranks) == 0:
            return "No funds identified for this peer group"
        else:
            top_performer = min(fund_ranks, key=lambda fund_name: fund_ranks[fund_name]["rank"])
            return top_performer
    
    # 4.3. Get Relevant Funds
    def get_relevant_funds(self, question):
        """
        Retrieves a list of relevant funds based on the given question.
        Parameters:
            question (str): The question to be used for retrieving relevant funds.
        Returns:
            str: A string containing the names of the relevant funds, separated by newlines.
        """
        top_k_str = self.infer_k_from_query_chain.invoke(question)["top_k"] # Infer top_k from the query
        if top_k_str.isdigit():
            top_k = int(top_k_str)
        else:
            top_k = 4 # The Qdrant default value for retrieval
        docs = self.get_score_retriever_with_k(top_k).invoke(question)
        context = ""
        for doc in docs:
            context += doc.metadata["fund_name"] + "\n"
        return context
    
    # 4.4. Count Funds in Knowledge Base
    def count_funds_in_knowledge_base(self, question) -> int:
        """
        Counts the number of funds in the knowledge base.
        Args:
            question (str): The question to be asked.
        Returns:
            int: The number of funds in the knowledge base.
        """
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
        # Assuming vectorstore.docs is a list or similar structure
        document_count = len(points)
        return document_count

    # 5. Define Chains
    def create_chains(self):
        """
        Creates different chains for the RagChain class.
        These chains are used for various tasks such as question answering, fund comparison, and performance analysis.
        """
        # Simple LLM chain without retrieval (for evaluation only)
        self.simple_llm = (
            PromptTemplate(template="Answer the question: {question}")
            | self.llm
            | RunnableLambda(lambda output: {"text": output})
        )

        # Subject classification chain
        self.subject_classification_chain = (
            prompts.subject_classification_template
            | self.llm
            | RunnableLambda(lambda output: {"text": output})
        )

        # Extract fund names chain
        self.extract_fund_names_chain = (
            prompts.extract_fund_names_template
            | self.llm
            | RunnableLambda(lambda output: {"text": output})
        )

        # Infer k from query chain
        self.infer_k_from_query_chain = (
            prompts.infer_k_from_query_template
            | self.llm
            | RunnableLambda(lambda output: {"top_k": output})
        )

        # Pass Question to Domain Knowledge Chain
        self.domain_specific_general_chain = (
            {
                "context": RunnableLambda(lambda inputs:
                                          self.gen_build_context(
                                              self.get_filtered_full_doc_retriever(inputs).invoke(inputs))),
                "question": RunnablePassthrough()
            }
            | prompts.general_template
            | self.llm
            | StrOutputParser()
        )

        # List all funds in knowledge base chain
        self.list_funds_in_knowledge_base_chain= (
            {
                "context": RunnableLambda(lambda inputs:
                                          self.gen_build_list_context(
                                              self.list_funds_in_knowledge_base(inputs))),
                "question": RunnablePassthrough()
            }
            | prompts.list_template
            | self.llm
            | StrOutputParser()
        )

        # Performance Description chain
        self.description_perf_chain = (
            {
            "context": RunnableLambda(lambda inputs: 
                self.build_perf_context(
                    self.get_filtered_full_doc_retriever(inputs).invoke(inputs),
                    self.subject_classification_chain.invoke(inputs)
                )
            ),
            "question": RunnablePassthrough()
            }
            | prompts.guided_description_perf_template
            | self.llm
            | StrOutputParser()
        )

        # Exposure Description chain
        self.description_exp_chain = (
            {
                "context": RunnableLambda(lambda inputs: 
                    self.exp_descriptive_process_documents(
                        self.get_filtered_full_doc_retriever(inputs).invoke(inputs),
                        self.subject_classification_chain.invoke(inputs)
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

        # Compare Simple Performance chain (only 2 funds)
        self.comp_simple_perf_chain = (
            {
                "context": RunnableLambda(lambda inputs: 
                    self.build_perf_context(
                        self.get_filtered_full_doc_retriever(inputs, top_k=2).invoke(inputs), # top_k = 2
                        self.subject_classification_chain.invoke(inputs)
                    )
                ),
            "question": RunnablePassthrough()
            }
            | prompts.compare_funds_perf_template
            | self.llm
            | StrOutputParser()
        )

        # Compare Peer Performance chain (identify Peer Group)
        self.comp_peer_perf_chain = (
            {
                "context": RunnableLambda(lambda inputs: 
                    self.compare_peer_performance_context(
                        self.get_score_retriever().invoke(inputs), # identify the peers
                        self.extract_fund_names_chain.invoke(inputs) # identify the fund in the query
                    )
                ),
            "question": RunnablePassthrough()
            }
            | RunnableLambda(lambda inputs: {
                "fund_name": inputs["context"]["fund_name"],
                "peers": inputs["context"]["peers"],
                "avg_1yr_return": inputs["context"]["avg_1yr_return"],
                "score_universe_rank": inputs["context"]["score_universe_rank"]
            })
            | prompts.analyze_peer_perf_score_template  # Step 2e: Formulate the query to find similar funds
            | self.llm  # Step 2f: Pass the query to the LLM to get the similar funds
            | StrOutputParser()  # Step 2g: Parse the output to get the final results
        )

        # Simple Exposure Comparison chain (only 2 funds)
        self.comp_simple_exp_chain = (
            {
                "context": RunnableLambda(lambda inputs: 
                    self.build_exp_context(
                        self.get_filtered_full_doc_retriever(inputs, top_k=2).invoke(inputs),
                        self.subject_classification_chain.invoke(inputs)
                    )
                ),
            "question": RunnablePassthrough()
        }
            | prompts.compare_funds_exp_template
            | self.llm
            | StrOutputParser()
        )

        # Compare Peer Exposure chain (identify Peer Group)
        self.comp_peer_exp_chain = (
            {
                "context": RunnableLambda(lambda inputs: 
                    self.compare_peer_exp_context(
                        self.get_score_retriever().invoke(inputs), # identify the peers
                        self.extract_fund_names_chain.invoke(inputs) # identify the fund in the query
                    )
                ),
            "question": RunnablePassthrough()
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

    # Define wrapping functions of chains
    def execute_domain_specific_general_chain(self, question):
        result = self.domain_specific_general_chain.invoke(question)
        return result
    
    def execute_list_all_funds_in_knowledge_base_chain(self, question):
        result = self.list_funds_in_knowledge_base_chain.invoke(question)
        return result
    
    def execute_description_perf_chain(self, question):
        result = self.description_perf_chain.invoke(question)
        return result
    
    def execute_description_exp_chain(self, question):
        result = self.description_exp_chain.invoke(question)
        return result
    
    def execute_compare_simple_perf_chain(self, question):
        result = self.comp_simple_perf_chain.invoke(question)
        return result
    
    def execute_compare_peer_perf_chain(self, question):
        result = self.comp_peer_perf_chain.invoke(question)
        return result
    
    def execute_compare_simple_exp_chain(self, question):
        result = self.comp_simple_exp_chain.invoke(question)
        return result
    
    def execute_compare_peer_exp_chain(self, question):
        result = self.comp_peer_exp_chain.invoke(question)
        return result

    # 6. Define Tools
    def create_tools(self):
        """
        Creates a list of Tool objects with their respective names, functions, and descriptions.
        Returns:
            list: A list of Tool objects.
        """
        self.domain_specific_general_tool = Tool(
            name="Pass Question to Domain Knowledge",
            func=self.execute_domain_specific_general_chain,
            description="Provides answers to specific information requests about exposure and performance from the investment fund factsheets."
        )

        self.count_funds_in_knowledge_base_tool = Tool(
            name="Count Funds in Collection",
            func=self.count_funds_in_knowledge_base,
            description="Counts the number of funds in the investment fund universe knowledge base."
        )

        self.list_funds_in_knwoledge_base_tool = Tool(
            name="Pass Question to list all funds in knowledge base",
            func=self.execute_list_all_funds_in_knowledge_base_chain,
            description="Passes list question to the knowledgable LLM to lists funds in the investment fund universe knowledge base."
        )

        self.get_relevant_funds_tool = Tool(
            name="Get Relevant Funds",
            func=self.get_relevant_funds,
            description= "Use this tool to get a specified number of relevant funds from the knowledge base if the question does not explicitly specify a fund name."
        )

        self.execute_description_exp_chain_tool = Tool(
            name="Provide Description of Exposure",
            func=self.execute_description_exp_chain,
            description="Use this tool when the question specifically asks for a detailed sector, regional or asset exposure description"
        )

        self.execute_description_perf_chain_tool = Tool(
            name="Provide Description of Performance",
            func=self.execute_description_perf_chain,
            description="Use this tool when the question specifically asks for a detailed performance description."
        )

        self.execute_compare_simple_perf_chain_tool = Tool(
            name="Compare Performance",
            func=self.execute_compare_simple_perf_chain,
            description="Use this tool when the question specifically involves comparing the performance metrics of two funds"
        )

        self.execute_compare_peer_perf_chain_tool = Tool(
            name="Compare Peer Performance",
            func=self.execute_compare_peer_perf_chain,
            description="Use this tool to compare the performance of a fund to its peer group."
        )

        self.execute_compare_simple_exp_chain_tool = Tool(
            name="Compare Exposure",
            func=self.execute_compare_simple_exp_chain,
            description=" Use this tool when the question specifically involves comparing sector, regional or asset exposure of two funds."
        )

        # Deactivated
        # self.execute_compare_peer_exp_chain_tool = Tool(
        #     name="Compare Peer Exposure",
        #     func=self.execute_compare_peer_exp_chain,
        #     description="Use this tool to compare the sector, regional or asset exposure of a fund to its peer group."
        # )

        self.execute_identify_top_performer_tool = Tool(
            name="Identify Top Performer",
            func=self.identify_top_performer,
            description="Use this tool to identify the fund with best 1yr performance in its peer group."
        )

        self.execute_identify_worst_performer_tool = Tool(
            name="Identify Worst Performer",
            func=self.identify_worst_performer,
            description="Use this tool to identify the fund with worst 1yr performance in its peer group."
        )

        tools = [
            self.domain_specific_general_tool,
            self.count_funds_in_knowledge_base_tool,
            self.list_funds_in_knwoledge_base_tool,
            self.get_relevant_funds_tool,
            self.execute_description_exp_chain_tool,
            self.execute_description_perf_chain_tool,
            self.execute_compare_simple_perf_chain_tool,
            self.execute_compare_peer_perf_chain_tool,
            self.execute_compare_simple_exp_chain_tool,
            #self.execute_compare_peer_exp_chain_tool, DEACTIVATED
            self.execute_identify_top_performer_tool,
            self.execute_identify_worst_performer_tool
        ]
        return tools
    
    # 7. Define Agent Executor
    def execute_agent_executor(self, inputs: Dict[str, Any]) -> Any:
        """
        Executes the agent executor with the given inputs.
        Args:
            inputs (Dict[str, Any]): The inputs for the agent executor.
        Returns:
            Any: The result of the agent executor.
        Raises:
            None

        ChatGBT (Prompt Engineering) https://chatgpt.com/share/b1d70a50-7942-4a29-902e-38001732eb01
        """
        # Ensure the inputs include the required variables
        # Get the question or provide a default
        input_value = inputs.get('question', '')
        agent_scratchpad = ''  # Initialize scratchpad as needed
        prompt = PromptTemplate.from_template("""
        You are a highly efficient assistant with access to the following tools:
        {tools}
                                              
        Your task is to answer the question by selecting and using the appropriate tool.
                                              
        If the answer can be provided without using any external tools, respond directly.
        
        Question: the input question you must answer
        Thought: I will find the correct tool and use it to get the answer.
        Action: If a tool is needed, choose the appropriate action from [{tool_names}]; otherwise, respond directly.
        Action Input: The input to the tool (i.e., the question or any specific details)
        Observation: The result returned by the tool
        Thought: I now know the final answer
        Final Answer: The response provided by the tool, do not modify the answer provided by the tool.

        Begin!

        Question: {input}
        Action:  
        If the tool used is one of [Compare Performance, Compare Peer Performance, Compare Exposure, Provide Description of Exposure, Provide Description of Performance],
        If the tool is one of [Identify Top Performer, Identify Worst Performer], then call one of the tools [Provide Description of Exposure,Provide Description of Performance]
        return the response from the tool as the Final Answer. Otherwise, select another tool from {tool_names} and use it.
        If the question lacks explicit fund names, use the **Get Relevant Funds** tool first, then select the appropriate tool from {tool_names}.
        If the question specificaly asks to `compare` funds, use one of the **Compare** tools.
        Thought: {agent_scratchpad})
        """)

        # Create the React agent with the prompt
        react_agent = create_react_agent(
            self.llm, tools=self.create_tools(), prompt=prompt)

        # Create the AgentExecutor using the defined agent
        react_agent_executor = AgentExecutor(
            agent=react_agent,
            tools=self.create_tools(),
            verbose=True,
            handle_parsing_errors=True
        )

        # Execute the agent with inputs
        result = react_agent_executor(
            {"input": input_value, "agent_scratchpad": agent_scratchpad})

        #print("React Agent Result: ", result)
        return result

    def execute_domain_specific_chain(self, inputs: Dict[str, Any]) -> Any:
        """
        Execute the domain specific chain (used for evaluation only)
        """
        return self.domain_specific_general_chain.invoke(inputs["question"])
    
    def execute_simple_llm(self, inputs: Dict[str, Any]) -> Any:
        """
        Execute the simple LLM model (used for evaluation only)
        """
        return self.simple_llm.invoke(inputs["question"])