import streamlit as st
import time

from openai import OpenAI
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()

es_client = Elasticsearch("http://localhost:9200")

def elastic_search(index_name: str, query: str, filter_course: str, num_results: int=5):
    search_query = {
        "size": num_results,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": filter_course
                    }
                }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)
    result_docs = [hit['_source'] for hit in response['hits']['hits']]
    return result_docs

def build_user_prompt(query: str, search_result: str):
    user_prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

    context_template = """
S: {section}
Q: {question}
A: {text}
""".strip()

    context= ""
    
    for doc in search_result:
        context += f"{context_template.format(section=doc['section'], question=doc['question'], text=doc['text'])}\n\n"

    user_prompt = user_prompt_template.format(question=query, context=context).strip()
    return user_prompt


def llm(user_prompt: str, model: str="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages= [
            {"role": "user", "content": user_prompt},
        ]
    )
    return response.choices[0].message.content


def rag_elastic(index_name: str, query: str, filter_course: str, num_results: int, model: str, history):
    # fields_list = [field.strip() for field in fields.split(',')]
    search_result = elastic_search(index_name, query, filter_course, num_results)
    user_prompt = build_user_prompt(query, search_result)
    answer = llm(user_prompt, model)
    return answer

# Inicializar el historial de chat si no existe
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


if 'params' not in st.session_state:
    st.session_state.params = {
        'index_name': 'homework-course',
        # 'fields': 'question^3,text,section',
        'filter_course': 'machine-learning-zoomcamp',
        'num_results': 1,
        'model': 'phi3'
    }

# Inicializar el estado del campo de entrada
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''

st.title("RAG Demo con Streamlit")


# Barra lateral para configuración
st.sidebar.header("Configuración del RAG")

# Lista de índices (ejemplo)
index_options = ['homework-course', 'another-index', 'third-index']

st.session_state.params['index_name'] = st.sidebar.selectbox("Índice", index_options, index=index_options.index(st.session_state.params['index_name']))

# st.session_state.params['fields'] = st.sidebar.text_input("Campos (separados por coma)", value=st.session_state.params['fields'])


# Lista de cursos (ejemplo)
course_options = ['data-engineering-zoomcamp', 'machine-learning-zoomcamp', 'mlops-zoomcamp', '']
st.session_state.params['filter_course'] = st.sidebar.selectbox("Curso", course_options, index=course_options.index(st.session_state.params['filter_course']))

st.session_state.params['num_results'] = st.sidebar.number_input("Número de resultados", min_value=1, max_value=10, value=st.session_state.params['num_results'])

# Lista de modelos (ejemplo)
model_options = ['phi3', 'gpt-3.5-turbo', 'gpt-4o']
st.session_state.params['model'] = st.sidebar.selectbox("Modelo", model_options, index=model_options.index(st.session_state.params['model']))


if st.session_state.params['model'] == "phi3":  
    client = OpenAI(
        base_url='http://localhost:11434/v1/',
        api_key='ollama',
    )
else:
    client = OpenAI()


# Área para mostrar el historial del chat
chat_container = st.container()

# Función para manejar el envío de la pregunta
def handle_submit():
    if st.session_state.user_input:
        query = st.session_state.user_input
        # Obtener respuesta
        response = rag_elastic(query=query,
                               history=st.session_state.chat_history,
                            #    index_name=index_name,
                            #    fields=fields, 
                            #    filter_course=filter_course, 
                            #    num_results=num_results, 
                            #    model=model,
                               **st.session_state.params)
        
        # Agregar la pregunta y respuesta al historial con timestamp
        timestamp = time.time()
        st.session_state.chat_history.append(("Usuario", query, timestamp))
        st.session_state.chat_history.append(("RAG", response, timestamp + 0.1))
        
        # Limpiar el campo de entrada
        st.session_state.user_input = ''
    else:
        st.write("Por favor, introduce una pregunta.")

# Campo de entrada para la nueva pregunta
st.text_input("Introduce tu pregunta:", key="user_input", on_change=handle_submit)

# Botón para enviar la pregunta
st.button("Enviar", on_click=handle_submit)


# Mostrar el historial del chat
with chat_container:
    for role, message, timestamp in st.session_state.chat_history:
        if role == "Usuario":
            st.text_input("Usuario:", value=message, key=f"user_{timestamp}", disabled=True)
        else:
            st.text_area("RAG:", value=message, key=f"rag_{timestamp}", disabled=True)


# Botón para limpiar el historial
if st.button("Limpiar historial"):
    st.session_state.chat_history = []
    st.experimental_rerun()


st.sidebar.header("Sobre esta demo")
st.sidebar.write("Esta es una demostración simple de cómo se podría implementar una interfaz para un sistema RAG utilizando Streamlit.")
