import os
import string
from langchain.docstore.document import Document
"""
This module defines several classes related to knowledge bases for financial documents.
- `DocumentsKnowledgeBase`: Represents a knowledge base that stores a collection of documents.
- `SingleDocumentKnowledgeBase`: Represents a knowledge base that aggregates multiple documents into a single document.
- `SectionedDocumentKnowledgeBase`: Represents a knowledge base that splits documents into sections based on tables.
- `InvObjKnowledgeBase`: Represents a knowledge base that extracts investment objective information from documents.
Each class provides methods to access and manipulate the documents in the knowledge base.
"""

# Import custom modules
import finbot.data_processing.full_processing as fp
import finbot.utils as utils

class DocumentsKnowledgeBase:
    """
    Represents a knowledge base for documents.
    Methods:
    - get_documents: Returns the list of documents in the knowledge base.
    - get_length: Returns the number of documents in the knowledge base.
    - format_content_for_documents_kb: Formats unstructured and structured text into a single document content.
    - build_knowledge_base: Builds the knowledge base by processing PDF files in a given directory.
    """
    def __init__(self): #, documents: list[Document]):
        self.documents = None
    
    def get_documents(self):
        """
        Returns the list of documents in the knowledge base.
        Returns:
            list: The list of documents in the knowledge base.
        """
        return self.documents
    
    def get_length(self):
        """
        Returns the length of the documents list.
        :return: The length of the documents list.
        :rtype: int
        """
        return len(self.documents)
    
    def format_content_for_documents_kb(self, unstructured_text: string, structured_text: string):
        """
        Formats the content for documents in the knowledge base.
        Args:
            unstructured_text (string): The unstructured text to be included in the document.
            structured_text (string): The structured text to be included in the document.
        Returns:
            string: The formatted content for the documents in the knowledge base.
        """
        full_text = "This is an investment fund factsheet for one fund containing the following fund information:\n"
        fund_information = utils.add_space_before_lines(unstructured_text)
        table_information = utils.add_space_before_lines(structured_text)
        full_text = full_text + fund_information + '\n\n' + table_information
        return full_text
    
    def build_knowledge_base(self, pdf_directory):
        """
        Builds the knowledge base by processing PDF files in the specified directory.
        Args:
            pdf_directory (str): The directory path containing the PDF files.
        Returns:
            list: A list of Document objects representing the processed PDF files.
        """
        documents = []
        for idx_doc, filename in enumerate(os.listdir(pdf_directory)):
            if filename.endswith('.pdf'):
                pdf_file = os.path.join(pdf_directory, filename)
                print(f"Processing {filename} ({idx_doc+1}/{len(os.listdir(pdf_directory))})")
                information = fp.pdf_file_processing_pipeline(pdf_file)
                if information is None:
                    print(f"Error loading the file {filename}")
                    continue
                print("Finished processing the file")
                doc_content = self.format_content_for_documents_kb(information['fund']['text'], information['tables']['text'])
                doc_metadata = {**information['fund']['metadata'], **information['tables']['metadata']}
                doc = Document(page_content=doc_content, metadata=doc_metadata)
                documents.append(doc)
                print(f"Loaded {filename} ({idx_doc+1}/{len(os.listdir(pdf_directory))})")
        print(f"There are {len(documents)} documents in the directory")
        self.documents = documents
        return self.documents
    
class SingleDocumentKnowledgeBase:
    """
    A class representing a single document knowledge base.
    Args:
        documents_kb (DocumentsKnowledgeBase): The source documents knowledge base.
    Attributes:
        src_documents (DocumentsKnowledgeBase): The source documents knowledge base.
        document (Document): The aggregated document.
    Methods:
        aggregate_documents: Aggregates the content and metadata of the source documents.
        get_document: Returns the aggregated document.
        get_metadata: Returns the metadata of the aggregated document.
        get_content: Returns the content of the aggregated document.
        get_length: Returns the length of the aggregated document.
    
    ChatGBT (implementation of inheritance stucture): https://chatgpt.com/share/1581b871-df48-4793-b8e4-deaa2f117903
    """

    def __init__(self, documents_kb: DocumentsKnowledgeBase):
        self.src_documents = documents_kb
        self.document = self.aggregate_documents()

    def aggregate_documents(self):
        """
        Aggregates the content and metadata of all source documents.
        """
        all_content = []
        all_metadata = {}

        for doc in self.src_documents.get_documents():
            # Aggregate all content and metadata
            all_content.append(doc.page_content)
            fund_name = doc.metadata['fund_name']
            all_metadata[fund_name] = doc.metadata

        aggregated_content = '\n\n'.join(all_content)
        aggregated_metadata = {'all_funds': all_metadata}

        return Document(page_content=aggregated_content, metadata=aggregated_metadata)

    def get_document(self):
        """
        Returns the document associated with the knowledge base.
        """
        return self.document
    
    def get_metadata(self):
        """
        Returns the metadata of the document.
        """
        return self.document.metadata
    
    def get_content(self):
        """
        Returns the page content of the document.
        """
        return self.document.page_content
    
    def get_length(self):
        """
        Returns the length of the document.
        """
        return 1 if isinstance(self.document, Document) else 0

    
class SectionedDocumentKnowledgeBase:
    """
    A class representing a sectioned document knowledge base.
    Attributes:
        TABLE_KEYS (list): A list of keys representing different sections in the document.
        src_documents (DocumentsKnowledgeBase): The source documents knowledge base.
        documents (list): A list of Document objects representing the chunked documents.
    Methods:
        __init__(self, documents_kb: DocumentsKnowledgeBase): Initializes the SectionedDocumentKnowledgeBase object.
        get_documents(self): Returns the list of chunked documents.
        get_length(self): Returns the number of chunked documents.
        split_content(self, content): Splits the content into fund summary and tables summary.
        split_metadata(self, metadata): Splits the metadata into fund metadata and tables metadata.
        chunk_tables_by_sections(self, text, verbose=False): Chunks the tables summary into sections.
        format_chunk_for_documents_sectioned(self, section_text: string, section_name: string, fund_name: string): Formats the section text for the chunked documents.
        chunk_documents(self): Chunks the source documents into fund summary and table sections.
    ChatGBT (implementation of inheritance stucture): https://chatgpt.com/share/1581b871-df48-4793-b8e4-deaa2f117903
    """

    TABLE_KEYS = ["Cumulative Performance", "Discrete Performance", "Asset Allocation", "Region Allocation", "Sector Allocation", "Largest Holdings", "Largest Holdings 2"]

    def __init__(self, documents_kb: DocumentsKnowledgeBase):
        self.src_documents = documents_kb
        self.documents = self.chunk_documents()
    
    def get_documents(self):
        """
        Returns the documents stored in the knowledge base.
        """
        return self.documents

    def get_length(self):
        """
        Returns the length of the documents list.
        """
        return len(self.documents)
    
    def split_content(self, content):
        """
        Splits the given content into fund summary and tables summary.
        Parameters:
        - content (str): The content to be split.
        Returns:
        - fund_summary (str): The extracted fund information from the content.
        - tables_summary (str): The extracted table information from the content.
        """
        parts = content.split('\n\n', 1)
        fund_summary = parts[0] if len(parts) > 0 else '' # extract the fund information from doc.page_content
        tables_summary = parts[1] if len(parts) > 1 else '' # extract the table information from doc.page_content
        #print("Fund summary: ", fund_summary)
        #print("Tables summary: ", tables_summary)
        return fund_summary, tables_summary

    def split_metadata(self, metadata):
        """
        Splits the given metadata into fund metadata and tables metadata.
        Parameters:
        - metadata (dict): The metadata dictionary to be split.
        Returns:
        - fund_metadata (dict): The fund metadata extracted from the given metadata.
        - tables_metadata (dict): The table metadata extracted from the given metadata.
        """
        fund_metadata = {k: v for k, v in metadata.items() if k not in self.TABLE_KEYS} # extract the fund metadata from doc.metadata
        tables_metadata = {k: v for k, v in metadata.items() if k in self.TABLE_KEYS} # extract the table metadata from doc.metadata
        return fund_metadata, tables_metadata
    
    def chunk_tables_by_sections(self, text, verbose = False):
        """
            Split the given text into sections based on newlines.
            Args:
                text (str): The text to be split into sections.
                verbose (bool, optional): If True, print the chunks for verification. Defaults to False.
            Returns:
                list: A list of sections, where each section is a string.
        ChatGBT: https://chatgpt.com/share/c2ffa745-948c-4091-af37-0db208d41c8a
        """
        # Split the text into sections based on newlines
        sections = text.split('\n \n') # take exactly as it shows in the table_summary
        # Remove any extra spaces or unnecessary formatting
        sections = [section.strip() for section in sections if section.strip()]

        # Print chunks for verification
        if verbose:
            for i, section in enumerate(sections):
                print(f"Chunk {i + 1}: {section}\n")

        print("Number of sections", len(sections))
        return sections
    
    def format_chunk_for_documents_sectioned(self, section_text: string, section_name: string, fund_name: string):
        """
        Formats a chunk of text for the documents section of an investment fund factsheet.
        Parameters:
            section_text (string): The text content of the section.
            section_name (string): The name of the section.
            fund_name (string): The name of the investment fund.
        Returns:
            string: The formatted text for the documents section.
        """
        full_text = f"This is the {section_name} section of the investment fund factsheet corresponding to {fund_name}:\n"
        full_text = full_text + section_text
        return full_text
    
    def chunk_documents(self):
        """
        Chunk the source documents into smaller sections and create document entries for each section.
        """
        documents = []
        for doc in self.src_documents.get_documents():
            fund_summary, tables_summary = self.split_content(doc.page_content)
            fund_metadata, tables_metadata = self.split_metadata(doc.metadata)
            print("Fund metadata: ", fund_metadata)
            print("Tables metadata: ", tables_metadata)
            # Create document entry for fund information
            summary_entry = Document(page_content=fund_summary, metadata=fund_metadata)
            documents.append(summary_entry)

            # Create document entries for each table section
            sections = self.chunk_tables_by_sections(tables_summary)
            section_names = list(tables_metadata.keys())
            print("Section names: ", section_names)
            for idx, section_text in enumerate(sections):
                section_text = self.format_chunk_for_documents_sectioned(section_text, section_names[idx], fund_metadata['fund_name'])
                print(f"Section text for {fund_metadata['fund_name']}")
                print(section_text)
                print("Loaded section: ", section_names[idx])
                print('-----------------------------------------------------')
                section_metadata = {'section_name': section_names[idx], 'table': tables_metadata[section_names[idx]]}
                # Remove investment objective metadata from the tables metadata (only for the chunked sections)
                fund_metadata.pop('inv_obj', None)
                print("Fund metadata after removing inv_obj: ", fund_metadata)
                full_metadata = {**section_metadata, **fund_metadata}
                section_entry = Document(page_content=section_text, metadata=full_metadata)
                documents.append(section_entry)
        return documents

class InvObjKnowledgeBase:
    """
    A class representing an investment objective knowledge base.
    Attributes:
        src_documents (DocumentsKnowledgeBase): The source documents knowledge base.
        documents (list): The built documents.
    Methods:
        get_documents(): Returns the built documents.
        get_length(): Returns the length of the built documents.
        build_documents(): Builds the documents based on the source documents.
    ChatGBT (implementation of inheritance stucture): https://chatgpt.com/share/1581b871-df48-4793-b8e4-deaa2f117903
    """

    def __init__(self, documents_kb: DocumentsKnowledgeBase):
        self.src_documents = documents_kb
        self.documents = self.build_documents()
    
    def get_documents(self):
        """
        Returns the documents stored in the knowledge base.
        """    
        return self.documents

    def get_length(self):
        """
        Returns the length of the documents list.
        """
        return len(self.documents)
    
    def build_documents(self):
        """
        Builds a list of documents based on the source documents.
        """
        documents = []
        for doc in self.src_documents.get_documents():
            doc_entry = Document(page_content=doc.metadata["inv_obj"], metadata={"fund_name": doc.metadata["fund_name"], "asset_class": doc.metadata["asset_class"]})
            documents.append(doc_entry)
        return documents