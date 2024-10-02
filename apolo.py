import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from notion_client import Client
from langchain.embeddings import OllamaEmbeddings  # Importando Ollama para embeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama  # Importando Ollama para o modelo de linguagem
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

# Carrega variáveis de ambiente e chaves de acesso
_ = load_dotenv(find_dotenv())

# Token de integração privada do Notion
notion_token = os.getenv("NOTION_TOKEN") # Substitua pelo seu token de integração privada
notion = Client(auth=notion_token)

# ID da página principal
page_id = "b04b3df810444abd9c1dbd02298f1684"  # ID da sua página de Base de Conhecimento


def process_block(block):
    """Processa um bloco do Notion e retorna um documento."""
    # Ignorar blocos vazios
    if block['type'] == 'paragraph' and not block['paragraph']['rich_text']:
        return None

    content = None
    # Mapeia os tipos de bloco para conteúdo
    if block['type'] == 'child_page':
        return Document(page_content=block['child_page']['title'], metadata={"type": "Subpágina"})
    elif block['type'] in ['heading_1', 'heading_2']:
        heading_type = 'Heading 1' if block['type'] == 'heading_1' else 'Heading 2'
        if 'text' in block[block['type']] and block[block['type']]['text']:
            content = ''.join([text['text']['content'] for text in block[block['type']]['text'] if 'text' in text])
            return Document(page_content=content, metadata={"type": heading_type})
    elif block['type'] == 'code':
        if 'text' in block['code'] and block['code']['text']:
            content = "```\n" + ''.join(
                [text['text']['content'] for text in block['code']['text'] if 'text' in text]) + "\n```"
            return Document(page_content=content, metadata={"type": "Code"})
    elif block['type'] == 'paragraph':
        if 'text' in block['paragraph'] and block['paragraph']['text']:
            content = ''.join([text['text']['content'] for text in block['paragraph']['text'] if 'text' in text])
            return Document(page_content=content, metadata={"type": "Conteúdo"})

    return None


def load_notion_subpages(subpage_id):
    """Carrega documentos de uma subpágina do Notion."""
    try:
        subpage_content = notion.blocks.children.list(subpage_id)
        documents = []

        for block in subpage_content['results']:
            result = process_block(block)
            if result:
                documents.append(result)

            # Recursão para subpáginas
            if block['type'] == 'child_page':
                child_id = block['id']
                child_documents = load_notion_subpages(child_id)
                documents.extend(child_documents)

        return documents
    except Exception as e:
        st.error(f"Erro ao acessar a subpágina: {e}")
        return []


@st.cache_data
def load_notion_content(page_id):
    """Carrega conteúdo da página principal do Notion."""
    try:
        page_content = notion.blocks.children.list(page_id)
        documents = []

        if not page_content['results']:
            return documents

        for block in page_content['results']:
            result = process_block(block)
            if result:
                documents.append(result)

            if block['type'] == 'child_page':
                child_id = block['id']
                child_documents = load_notion_subpages(child_id)
                documents.extend(child_documents)

        return documents
    except Exception as e:
        st.error(f"Erro ao acessar a página principal: {e}")
        return []


# Carrega documentos da página principal
with st.spinner("Carregando dados..."):
    documents = load_notion_content(page_id)

# Verifica se os documentos foram carregados
if not documents:
    st.error("Nenhum documento encontrado.")
else:
    st.write(f"{len(documents)} documentos encontrados com sucesso.")

    # Cria embeddings e FAISS VectorStore usando Ollama
    embeddings = OllamaEmbeddings()  # Usando Ollama para gerar embeddings
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Cria um retriever a partir dos embeddings
    retriever = vectorstore.as_retriever()

    # Cria o chain de perguntas e respostas usando o LLM da Ollama
    llm = Ollama(model="llama3.1:8b-instruct-q4_K_S", base_url="http://localhost:11434")
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever)

    # Interface de chat no Streamlit
    st.title("Apollum - Chat de Conhecimento")

    # Histórico de perguntas e respostas
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Exibe o histórico do chat
    for chat in st.session_state['chat_history']:
        st.write(f"**Você:** {chat['question']}")
        st.write(f"**Apollum:** {chat['answer']}")

    # Caixa de input para perguntas
    question = st.text_input("Digite sua pergunta:")

    if question:
        # Executa a pergunta no modelo de linguagem
        chat_history = [(chat['question'], chat['answer']) for chat in st.session_state['chat_history']]
        result = qa_chain.run({"question": question, "chat_history": chat_history})

        # Adiciona o resultado no histórico
        st.session_state['chat_history'].append({
            "question": question,
            "answer": result
        })

        # Atualiza a página com a nova resposta
        st.rerun()  # Manter rerun para atualizar a interface
