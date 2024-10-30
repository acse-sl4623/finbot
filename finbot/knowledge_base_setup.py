import os
import sys

# Add the root directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))#, '../../')))
"""
Builds different knowledge bases and creates vector stores for each knowledge base.

1. Build the knowledge base defined by each point representing an individual pdf document.
    - Parameters:
      - pdf_directory (str): The directory containing the PDF documents.
    - Returns:
      - docs (list): List of documents in the knowledge base.

2. Build the knowledge base defined by one point representing all the documents merged together.
    - Parameters:
      - main_kb (kb.DocumentsKnowledgeBase): The main knowledge base containing individual pdf documents.
    - Returns:
      - whole_kb (kb.SingleDocumentKnowledgeBase): The knowledge base representing all the documents merged together.

3. Build the knowledge base defined by each point representing an individual section of a pdf document.
    - Parameters:
      - main_kb (kb.DocumentsKnowledgeBase): The main knowledge base containing individual pdf documents.
    - Returns:
      - sec_kb (kb.SectionedDocumentKnowledgeBase): The knowledge base representing each section of a document.

4. Build the knowledge base defined by each point representing just the investment objective.
    - Parameters:
      - main_kb (kb.DocumentsKnowledgeBase): The main knowledge base containing individual pdf documents.
    - Returns:
      - obj_kb (kb.InvObjKnowledgeBase): The knowledge base representing only the investment objective.
"""

import finbot.knowledge_base as kb # noqa: F401
import finbot.vectorstore as vs # noqa: F401

# 1. Build the knowledge base defined by each point representing an individual pdf document
pdf_directory = 'factsheets/trustnet'
main_kb = kb.DocumentsKnowledgeBase()
docs = main_kb.build_knowledge_base(pdf_directory)
print(f"Number of documents: {len(docs)}", '\n')

collection_name = f'all_docs_kb_clean_inv'
vectorstore = vs.QdrantVectorStore(collection_name=collection_name)
vectorstore.create_vectorstore(collection_name, docs)
print(f"{collection_name} vectorstore created.")
vectorstore.inspect_collection_upload(collection_name)

# 2. Build the knowledge base defined by one point representing all the documents merged together
whole_kb = kb.SingleDocumentKnowledgeBase(main_kb)
print(f"Size of aggregated knowledge base {whole_kb.get_length()}")
print(f"Keys of all the funds metadata {whole_kb.get_metadata().keys()}", '\n')

collection_name = f'all_single_kb_clean_inv'
vectorstore = vs.QdrantVectorStore(collection_name=collection_name)
vectorstore.create_vectorstore(collection_name, whole_kb.get_document())
print(f"{collection_name} vectorstore created.")
vectorstore.inspect_collection_upload(collection_name)

# 3. Build the knowledge base defined by each point representing an individual section of a pdf document
sec_kb = kb.SectionedDocumentKnowledgeBase(main_kb)
print(f"Number of chunked documents {sec_kb.get_length()}")
print(f"Example document chunk: {sec_kb.get_documents()[0]}", '\n')

collection_name = f'all_sec_kb_clean_inv'
vectorstore = vs.QdrantVectorStore(collection_name=collection_name)
vectorstore.create_vectorstore(collection_name, sec_kb.get_documents())
print(f"{collection_name} vectorstore created.")
vectorstore.inspect_collection_upload(collection_name)

# 4. Build the knowledge base defined by each point representing just the investment objective
obj_kb = kb.InvObjKnowledgeBase(main_kb)
print(f"Number of objective documents {obj_kb.get_length()}")
print(f"Example objective document: {obj_kb.get_documents()[0]}", '\n')

collection_name = f'all_obj_kb_clean_inv'
vectorstore = vs.QdrantVectorStore(collection_name=collection_name)
vectorstore.create_vectorstore(collection_name, obj_kb.get_documents())
print(f"{collection_name} vectorstore created.")
vectorstore.inspect_collection_upload(collection_name)