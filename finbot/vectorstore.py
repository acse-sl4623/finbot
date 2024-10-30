from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from qdrant_client import models, QdrantClient
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

class QdrantVectorStore:
    """
    A class representing a vector store in Qdrant.
    Args:
        collection_name (str): The name of the collection.
    Attributes:
        API_KEY (str): The API key retrieved from environment variables.
        LLM_API_KEY (str): The LLM API key retrieved from environment variables.
        QDRANT_HOST (str): The Qdrant host URL.
        QDRANT_PORT (int): The Qdrant port number.
        QDRANT_URL (str): The Qdrant URL.
        client (QdrantClient): The Qdrant client.
        collection_name (str): The name of the collection.
        vectorstore (Qdrant): The Qdrant vector store.
        size (int): The size of the vector store.
    Methods:
        load_vectorstore(collection_name): Loads the vector store from an existing collection.
        create_vectorstore(collection_name, docs=None): Creates or loads the vector store.
        inspect_collection_upload(collection_name, verbose=False): Inspects the collection upload.
    """
    def __init__(self, collection_name):
        # Retrieve API keys from environment variables
        self.API_KEY = os.getenv('API_KEY')
        self.LLM_API_KEY = os.getenv('LLM_API_KEY')

        # Qdrant configuration
        self.QDRANT_HOST = '4cf1407b-51ba-45c1-8b40-1a345da0049a.us-east4-0.gcp.cloud.qdrant.io'
        self.QDRANT_PORT = 6333
        self.QDRANT_URL = f"https://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

        # Initialize the Qdrant client
        self.client = QdrantClient(host=self.QDRANT_HOST, port=self.QDRANT_PORT, api_key=self.API_KEY)
        self.collection_name = collection_name
        self.vectorstore = None
        self.size = None

    def load_vectorstore(self, collection_name):
        """
        Loads a vector store for the specified collection name.
        Parameters:
            collection_name (str): The name of the collection to load the vector store for.
        Returns:
            Qdrant: The loaded vector store.
        Raises:
            None
        """

        model_name = "sentence-transformers/all-mpnet-base-v2"
        embedding_model = HuggingFaceEmbeddings(model_name=model_name) 
        collections = self.client.get_collections()
        if collection_name in [collection.name for collection in collections.collections]:
            self.vectorstore = Qdrant.from_existing_collection(
                embedding=embedding_model,
                collection_name=collection_name,
                url=self.QDRANT_URL,
                api_key=self.API_KEY
            )
        return self.vectorstore

    # Split this in two methods: one for creating the collection and one for loading it
    def create_vectorstore(self, collection_name, docs = None): #, split = False):
        """
        Creates a vector store for the given collection name and documents.
        Args:
            collection_name (str): The name of the collection.
            docs (list[Document], optional): The list of documents to be added to the vector store. Defaults to None.
        Returns:
            Qdrant: The created vector store.
        Raises:
            None
        """
        # embeddings = OllamaEmbeddings(model="llama3")
        model_name = "sentence-transformers/all-mpnet-base-v2"  # Change this to the desired model
        embedding_model = HuggingFaceEmbeddings(model_name=model_name) 
        collections = self.client.get_collections()
        if collection_name not in [collection.name for collection in collections.collections]:
            print(f"Collection {collection_name} created")

            if isinstance(docs, Document):
                docs = [docs]
            
            self.vectorstore = Qdrant.from_documents(
                    documents=docs, # documents must be a list of Document objects
                    embedding=embedding_model, # This is the embedding_model/ each page_content is re-embedded as per above model
                    url=self.QDRANT_URL,
                    api_key=self.API_KEY,
                    collection_name=collection_name,
                    )
        else:
            self.vectorstore = self.load_vectorstore(collection_name)
        return self.vectorstore
    
    def inspect_collection_upload(self, collection_name, verbose = False):
        """
        Inspects the collection upload in the vectorstore.
        Parameters:
            collection_name (str): The name of the collection to inspect.
            verbose (bool, optional): If True, prints the payload and vector for each point. Defaults to False.
        Returns:
            int: The size of the vectorstore.
        """

        scroll_response = self.client.scroll(
            collection_name=collection_name,
            limit=100, # Limit the number of points to retrieve (defaults to 10) so just put a very high number
            with_vectors=True, # Retrieve the vectors associated with the points
            with_payload=True  # Retrieve the payloads associated with the points
        )

        # print("Scroll response:", scroll_response)
        points = scroll_response[0] # if limit>nb of points then there is no response[1]
        if verbose:
            for point in points:
                print("Payload", point.payload) # we observe the page_content is in the payload
                print("Vector", point.vector,'\n') # we observe the vector is in the vector
        else:
            print(f"The vectorstore {collection_name} has the following attributes:")
            #print("Number of points: ", len(points)) This is not correct
            print("Embeddings shape: ", len(points[0].vector))
            #print("Metadata: ", points[0].payload['metadata'])
            self.size = len(points)
        return self.size