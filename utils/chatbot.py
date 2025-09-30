import os
from typing import List, Tuple
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

class ChatBot:
    """
    Class representing a chatbot with document retrieval, evaluation, and response generation.
    """

    def __init__(self):
        """
        Initialize the ChatBot with configuration from global variables.
        """
        self.log = ""
        from load_config import load_config_from_yaml
        #config_path = os.path.join('..', 'configs', 'config.yaml')
        config = load_config_from_yaml()
        
        # Direct assignment without intermediate variables
        self.directories = config['directories']
        self.retrieval_config = config['retrieval_config']
        self.llm_config = config['llm_config']
        
        # Initialize clients
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.embedding = OpenAIEmbeddings()

    def add_log(self, log_message: str) -> None:
        """Add a message to the log."""
        self.log += log_message + "\n"

    def get_log(self) -> str:
        """Get the current log content."""
        return self.log

    def clear_log(self) -> None:
        """Clear the log."""
        self.log = ""

    # Evaluating search results in the vectordb using LLM
    def evaluate_retrieved_documents(self, user_query: str, documents: List, temperature: float = 0.0) -> Tuple[bool, List]:
        """
        Evaluate if retrieved documents are relevant to the user query using an LLM.

        Parameters:
            user_query: The original user question
            documents: List of retrieved documents
            temperature: Temperature for evaluation LLM (use low temperature for evaluation)

        Returns:
            Tuple: (are_documents_relevant, filtered_documents)
        """
        try:
            # Prepare document content for evaluation
            doc_contents = "\n\n".join([
                f"Document {i+1}:\n{doc.page_content[:1000]}..." if len(doc.page_content) > 1000
                else f"Document {i+1}:\n{doc.page_content}"
                for i, doc in enumerate(documents)
            ])

            evaluation_prompt = f"""
            You are a document relevance evaluator. Your task is to determine if the retrieved documents are relevant to answering the user's question.

            USER QUESTION: {user_query}

            RETRIEVED DOCUMENTS:
            {doc_contents}

            INSTRUCTIONS:
            1. Analyze each document's content against the user question
            2. Determine if the documents contain information that can help answer the question
            3. If documents are relevant, respond with "RELEVANT: [brief explanation]"
            4. If documents are not relevant, respond with "NOT_RELEVANT: [brief explanation]"
            5. Also identify which specific document numbers (1-{len(documents)}) are most relevant

            Respond in this exact format:
            VERDICT: RELEVANT or NOT_RELEVANT
            EXPLANATION: [your explanation]
            RELEVANT_DOCS: [comma-separated list of relevant document numbers, or "none"]
            """

            evaluation_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using a cheaper/faster model for evaluation
                messages=[
                    {"role": "system", "content": "You are a precise document relevance evaluator."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=temperature,
                #max_tokens=500
            )

            eval_content = evaluation_response.choices[0].message.content
            self.add_log(f"Document Evaluation Result:\n{eval_content}")

            # Parse the response
            lines = eval_content.split('\n')
            verdict = None
            relevant_docs = []

            for line in lines:
                if line.startswith('VERDICT:'):
                    verdict = 'RELEVANT' in line.upper()
                elif line.startswith('RELEVANT_DOCS:'):
                    docs_part = line.split(':')[1].strip()
                    if docs_part.lower() != 'none':
                        relevant_docs = [int(d.strip()) for d in docs_part.split(',') if d.strip().isdigit()]

            # Filter documents based on evaluation
            if verdict and relevant_docs:
                filtered_docs = [documents[i-1] for i in relevant_docs if 1 <= i <= len(documents)]
                self.add_log(f"Relevant documents identified: {relevant_docs}")
                return True, filtered_docs
            elif verdict:
                # If generally relevant but no specific docs mentioned, keep all
                self.add_log("Documents generally relevant, keeping all")
                return True, documents
            else:
                self.add_log("Documents deemed not relevant")
                return False, []

        except Exception as e:
            self.add_log(f"Error in document evaluation: {str(e)}")
            # If evaluation fails, assume documents are relevant as fallback
            return True, documents

    def respond(self, chatbot: List, message: str, temperature: float = None) -> Tuple:
        """
        Generate a response to a user query with retrieval evaluation.
        """
        try:
            if temperature is None:
                temperature = self.llm_config["temperature"]

            # Check if VectorDB exists
            if (not os.path.exists(self.directories['persist_directory']) or len(os.listdir(self.directories['persist_directory'])) == 0):
                chatbot.append((message, "VectorDB does not exist or is empty. Please first initialize the VectorDB"))
                self.add_log("VectorDB does not exist")
                return "", chatbot, None

            # Load VectorDB and retrieve documents
            vectordb = Chroma(
                persist_directory=self.directories['persist_directory'],
                embedding_function=self.embedding
            )

            initial_docs = vectordb.similarity_search(message, k=self.retrieval_config['k'])
            self.add_log(f"Initially retrieved {len(initial_docs)} documents for evaluation")
            if not initial_docs:
                chatbot.append((message, "No documents found for your query."))
                return "", chatbot, "No documents found"

            # Evaluate document relevance
            are_relevant, filtered_docs = self.evaluate_retrieved_documents(
                user_query=message,
                documents=initial_docs,
                temperature=0.0
            )

            if not are_relevant or not filtered_docs:
                # If no relevant documents found
                no_docs_response = "I couldn't find relevant information in the documents to answer your question. Please try rephrasing or ask about something else."
                chatbot.append((message, no_docs_response))
                self.add_log("No relevant documents found - provided fallback response")
                return "", chatbot, "No relevant documents found."

            self.add_log(f"After evaluation: using {len(filtered_docs)} relevant documents")

            # Format the relevant content for final response
            retrieved_content = '\n'.join([
                f'Relevant Document {i+1}:\n{doc.page_content}\n'
                for i, doc in enumerate(filtered_docs)
            ])

            # Final prompt with relevant documents only
            question = f"# User new question:\n{message}"
            prompt = f"{retrieved_content}{question}"

            self.add_log(f"Final prompt prepared with {len(filtered_docs)} relevant documents")

            response = self.client.chat.completions.create(
                model=self.llm_config["engine"],
                messages=[
                    {"role": "system", "content": self.llm_config["llm_system_role"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )

            response_content = response.choices[0].message.content
            chatbot.append((message, response_content))

            # Log token usage
            if hasattr(response, 'usage') and response.usage:
                self.add_log('\nToken statistics for final response:')
                self.add_log(f'Prompt tokens: {response.usage.prompt_tokens}')
                self.add_log(f'Completion tokens: {response.usage.completion_tokens}')
                self.add_log(f'Total tokens: {response.usage.total_tokens}')

            return "", chatbot, retrieved_content

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.add_log(error_msg)
            chatbot.append((message, error_msg))
            return "", chatbot, error_msg