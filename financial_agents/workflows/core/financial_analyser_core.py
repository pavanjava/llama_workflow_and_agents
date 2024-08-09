from llama_index.core.node_parser import UnstructuredElementNodeParser
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import SimpleDirectoryReader, Settings, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
import os
import pickle


class FinancialAnalyserCore:
    def __init__(self, financial_report_file: str):
        llm = Ollama(model='llama3.1', request_timeout=300)
        embed_model = OllamaEmbedding(model_name='all-minilm:33m')
        text_parser = SentenceSplitter(chunk_size=128, chunk_overlap=100)
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        self.financial_report_file = financial_report_file

        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.transformations = [text_parser]
        Settings.callback_manager = callback_manager

        reader = SimpleDirectoryReader(input_files=[f"data/{financial_report_file}"])
        self.docs_2023 = reader.load_data(show_progress=True)
        self.base_nodes_2023 = None

        self.node_mappings_2023 = None
        self.retriever = None
        self._pre_process()

    def _pre_process(self):

        node_parser = UnstructuredElementNodeParser()
        pickle_file = f"./{self.financial_report_file.rstrip('.pdf')}.pkl"
        if not os.path.exists(pickle_file):
            raw_nodes_2023 = node_parser.get_nodes_from_documents(self.docs_2023)
            pickle.dump(raw_nodes_2023, open(pickle_file, "wb"))
        else:
            raw_nodes_2023 = pickle.load(open(pickle_file, "rb"))

        self.base_nodes_2023, self.node_mappings_2023 = node_parser.get_base_nodes_and_mappings(
            raw_nodes_2023
        )
        self._index_in_vector_store()

    def _index_in_vector_store(self):
        # Create a local Qdrant vector store
        client = qdrant_client.QdrantClient(url="http://localhost:6333/", api_key="th3s3cr3tk3y")
        vector_store = QdrantVectorStore(client=client, collection_name=f"{self.financial_report_file.strip('.pdf')}")

        # construct top-level vector index + query engine
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex(nodes=self.base_nodes_2023, storage_context=storage_context,
                                        transformations=Settings.transformations, embed_model=Settings.embed_model)

        self.retriever = vector_index.as_retriever(similarity_top_k=5)

    def retriever_query_engine(self):
        recursive_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": self.retriever},
            node_dict=self.node_mappings_2023,
            verbose=True,
        )
        query_engine = RetrieverQueryEngine.from_args(recursive_retriever)
        return query_engine

# GAAP-based gross profit and gross margin
# What was the GAAP-based net income attributable to OpenText
# What was the Reconciliation of selected GAAP-based measures to Non-GAAP-based "
#                               "measures for the nine months ended March 31, 2023
