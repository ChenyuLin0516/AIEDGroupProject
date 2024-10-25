from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma

import chainlit as cl
import os

# Initialize embeddings and chat model
google_api_key = os.getenv("GOOGLE_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=google_api_key
)

groq_api_key = os.getenv("GROQ_API_KEY")
chat_model = ChatGroq(
    model="mixtral-8x7b-32768", 
    api_key=groq_api_key,
    streaming=True
)
#Test API key
#GROQ_API_KEY=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
#GOOGLE_API_KEY=AIzaSyCUU8pGrORw3LX9AJ0BciRozCgJX9K-T7k

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="# Welcome to PrepMaster! 🚀🤖\nHere, you will be helped by a team of virtual experts to better prepare for the weekly AI Application course.").send()    
    #need to modify

    #### Ask for the PDF upload
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Hi there! Very happy to meet you here. Can you upload the materials of this week?",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    path = file.path

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Load and process the PDF
    loader = PyMuPDFLoader(path)
    loaded_pdf = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1000)
    texts = text_splitter.split_documents(loaded_pdf)
    print("Chunking ready")

    # Use the previously defined embeddings
    docsearch = Chroma.from_documents(texts, embeddings)
    print("Embeddings Ready")

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

# Create agents for questioning and evaluation
    QuestioneGenerator = Agent(
        role='QuestioneGenerator',
        goal='Generate questions about the uploaded document to assess user understanding',
        backstory="You are an expert in generating questions based on the summary of the learning content of the week. You like to measure students’ understanding of the learning materials with the right type of question for specific learning content by comparing students’ performance in the pre-test and post-test.",
        tools=[],
        llm=chat_model
    )

    evaluator = Agent(
        role='Evaluator',
        goal='Evaluate the user’s answer and provide constructive feedback',
        backstory="You’re an expert in evaluating students’ prior knowledge  based on the answers in the pre-test and providing suggestions to Question Generator during the test. ",
        tools=[],
        llm=chat_model
    )

#Plz Add all the agents we need！
# Learning Content Analyst，Facilitator，Summary Expert 



    # Define tasks for each agent
    question_generating = Task(
        description="generate 3 questions based on the uploaded learning materials",
        expected_output="A list of 3 questions",
        agent=QuestioneGenerator,
    )

    pretest_evaluate = Task(
        description="evaluate the student’s knowledge based on the QA report",
        expected_output="an evaluation report with assessment summary of each question",
        agent=evaluator,
    )
    #Plz Add all the tasks we need！

    # Create a crew to handle the sequential execution of tasks
    crew = Crew(
        agents=[question_generating, evaluator],
        tasks=[question_generating, pretest_evaluate],
        verbose=True,
        process=Process.sequential  # Execute tasks in sequence
    )

    # Store crew and docsearch in user session for further use
    cl.user_session.set('crew', crew)
    cl.user_session.set('docsearch', docsearch)  # Store the document search for reference

    await cl.Message(content="Agents are ready! We can now start the Q&A based on the document content.").send()



@cl.on_message
async def main(message: cl.Message):
    crew = cl.user_session.get("crew")  
    docsearch = cl.user_session.get("docsearch")
    
    # Initialize the question-answer process with the input document content
    topic = message.content
    inputs = {'question': topic}

    # Trigger crew to generate a question and then evaluate the answer
    crew_output = crew.kickoff(inputs=inputs)

    # Extract results
    question_result = crew_output['question']
    evaluation_result = crew_output['evaluation']

    # Display the question and evaluation results
    await cl.Message(content=f"Question: {question_result}\nEvaluation: {evaluation_result}").send()