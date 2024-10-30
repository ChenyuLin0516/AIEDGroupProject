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

    # Define agents

    agents = {
        "LearningContentAnalyst": Agent(
        role='LearningContentAnalyst',
        goal='Analyze content from PDF and provide brief summary for context.',
        backstory="You’re an expert in analyzing content in the uploaded materials and summarizing them so that they are easily digestible.",
        tools=[],
        llm=chat_model
    ),
        "Evaluator": Agent(
        role='Evaluator',
        goal='Evaluate the user’s answer and provide constructive feedback',
        backstory="You’re an expert in evaluating students’ prior knowledge based on the answers in the pre-test and providing suggestions to the QuestionGenerator during the test.",
        tools=[],
        llm=chat_model
    ),
        "QuestionGenerator": Agent(
        role='QuestionGenerator',
        goal='Generate questions about the uploaded document to assess user understanding',
        backstory="You are an expert in generating questions based on the summary of the learning content for the week. You like to measure students' understanding of the learning materials with the right type of questions for specific learning content by comparing students' performance in the pre-test and post-test.",
        tools=[],
        llm=chat_model
    ),
        "Facilitator": Agent(
        role='Facilitator',
        goal='Facilitate conversation with user to encourage active thinking and aid their understanding',
        backstory="You’re an expert in facilitating learning through asking questions related to the summary of the learning content and students’ pre-test performance. You like to encourage students’ active thinking with questions. You normally ask questions for three rounds of conversation and then provide explanations and examples if students still don’t understand.",
        tools=[],
        llm=chat_model
    ),
        "SummaryExpert": Agent(
        role='SummaryExpert',
        goal='Summarize user conversation history and learning content',
        backstory="You’re an expert in summarizing the conversation history between students and the AI agents. Your summary incorporates the summary of the learning content with the record of students’ question answering. The summary can be supplemental materials for students to review later for their study.",
        tools=[],
        llm=chat_model
    )
}

# Define tasks for each agent in the desired workflow sequence
    tasks = [
    Task(
        description="Analyze the weekly learning materials and generate a summary.",
        expected_output="A summary of the weekly learning materials with key content.",
        agent=agents["LearningContentAnalyst"]
    ),
    Task(
        description="Generate pre-test questions based on the summary provided by LearningContentAnalyst.",
        expected_output="A list of 3 pre-test questions.",
        agent=agents["QuestionGenerator"]
    ),
    Task(
        description="Ask pre-test questions one by one to the user and record answers.",
        expected_output="A Q&A report including the user's answers.",
        agent=agents["QuestionGenerator"]
    ),
    Task(
        description="Evaluate the user's answers from pre-test Q&A and provide feedback for Facilitator.",
        expected_output="An evaluation report (not shown to the user).",
        agent=agents["Evaluator"]
    ),
    Task(
        description="Using Evaluator’s feedback, guide the user through the key learning points, allowing for user interaction.",
        expected_output="A report of the learning session with key content explanations.",
        agent=agents["Facilitator"]
    ),
    Task(
        description="Generate post-test questions based on the learning content covered.",
        expected_output="A list of 3 post-test questions.",
        agent=agents["QuestionGenerator"]
    ),
    Task(
        description="Ask post-test questions one by one and record user's answers.",
        expected_output="A Q&A report including user's answers.",
        agent=agents["QuestionGenerator"]
    ),
    Task(
        description="Evaluate post-test answers and create an assessment report (not shown to user).",
        expected_output="A post-test evaluation report.",
        agent=agents["Evaluator"]
    ),
    Task(
        description="Based on interaction history and post-test evaluation, generate a final learning report and notes for the user.",
        expected_output="A PDF summary of learning content and performance feedback.",
        agent=agents["Facilitator"]
    )
]
    # Create a crew to handle the sequential execution of tasks
    crew = Crew(
    agents=list(agents.values()),
    tasks=tasks,
    verbose=True,
    process=Process.sequential
)

    # Store crew and docsearch in user session for further use
    cl.user_session.set('crew', crew)
    cl.user_session.set('docsearch', docsearch)  # Store the document search for reference

@cl.on_message
async def main(message: cl.Message):
    crew = cl.user_session.get("crew")  
    docsearch = cl.user_session.get("docsearch")
    
    # Capture the input topic from the user's message
    topic = message.content
    inputs = {'question': topic}

    # Kickoff the crew task sequence
    crew_output = await crew.kickoff(inputs=inputs)

    # Retrieve output from each specific task as per your workflow
    summary_result = crew_output.get('LearningContentAnalyst', {}).get('summary')
    pretest_questions = crew_output.get('QuestionGenerator', {}).get('pre_test_questions')
    pretest_answers = crew_output.get('QuestionGenerator', {}).get('pre_test_answers')
    pretest_evaluation = crew_output.get('Evaluator', {}).get('evaluation')
    learning_content = crew_output.get('Facilitator', {}).get('content')
    posttest_questions = crew_output.get('QuestionGenerator', {}).get('post_test_questions')
    posttest_answers = crew_output.get('QuestionGenerator', {}).get('post_test_answers')
    posttest_evaluation = crew_output.get('Evaluator', {}).get('post_evaluation')
    final_summary = crew_output.get('Facilitator', {}).get('summary_report')

    # Display results sequentially or according to your requirements
    await cl.Message(content=f"Summary: {summary_result}").send()
    await cl.Message(content=f"Pre-test Questions: {pretest_questions}\nUser Answers: {pretest_answers}").send()
    await cl.Message(content=f"Evaluation Report (not shown to user): {pretest_evaluation}").send()
    await cl.Message(content=f"Learning Content: {learning_content}").send()
    await cl.Message(content=f"Post-test Questions: {posttest_questions}\nUser Answers: {posttest_answers}").send()
    await cl.Message(content=f"Post-test Evaluation (not shown to user): {posttest_evaluation}").send()
    await cl.Message(content=f"Final Summary Report: {final_summary}").send()

