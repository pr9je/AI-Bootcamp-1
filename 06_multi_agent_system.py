
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not API_KEY:
    raise ValueError("GROQ_API_KEY is missing from your .env file.")

llm = ChatGroq(
    model=MODEL,
    temperature=0,
    max_tokens=700,
    api_key=API_KEY,
)

parser = StrOutputParser()



def run_agent(role, instructions, question):
    """Run one specialist agent with a concise response limit."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{instructions}"),
        ("human", "{question}"),
    ])

    # Limit the amount of text sent into the model.
    compact_question = question[:3500]

    chain = prompt | llm | parser

    return chain.invoke({
        "instructions": (
            f"{instructions}\n\n"
            "Keep your response concise: maximum 250 words. "
            "Use short bullet points. Include only essential details."
        ),
        "question": compact_question,
    })


def main():
    question = input("Enter your HVAC VRF/VRV question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    print("\n--- AGENT 1: HVAC EXPERT ---")

    hvac_answer = run_agent(
        "HVAC Expert",
        (
            "You are an HVAC VRF/VRV learning expert. "
            "Explain concepts in beginner-friendly language with practical "
            "examples. Do not invent equipment specifications."
        ),
        question,
    )
    print(hvac_answer)

    print("\n--- AGENT 2: ENGINEERING ANALYST ---")

    engineering_answer = run_agent(
        "Engineering Analyst",
        (
            "You are an engineering analyst supporting HVAC learning. "
            "Identify relevant inputs, assumptions, and calculations. "
            "Never invent missing design data. State when information "
            "is insufficient for a reliable engineering calculation."
        ),
        f"Original question: {question}\n\n"
        f"HVAC expert's response:\n{hvac_answer}",
    )
    print(engineering_answer)

    print("\n--- AGENT 3: REVIEWER ---")

    review_input = (
        f"Original question: {question}\n\n",
        f"HVAC exper summary:\n{hvac_answer[:2500]}\n\n"
        "Review these responses. Identify important errors, missing inputs, "
        "unsupported assumptions, and safety concerns. "
        "Give a concise final answer in no more than 200 words."
    )

    review = run_agent(
        "Reviewer",
        (
            "You are a technical reviewer for HVAC VRF/VRV explanations. "
            "Be concise. Do not invent equipment specifications or calculations. "
            "Flag uncertainty and recommend qualified engineering review "
            "for actual equipment selection."
        ),
        review_input,
)

    print(review)

if __name__ == "__main__":
    main()
