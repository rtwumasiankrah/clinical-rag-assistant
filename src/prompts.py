"""Prompt templates for grounded clinical Q&A and LLM-as-judge evaluation.

The system prompt is deliberately strict: the model must answer *only* from
the retrieved context and say "I don't know" otherwise. In a clinical
setting, a refused answer is always safer than a hallucinated one.
"""

QNA_SYSTEM_MESSAGE = """
You are an assistant whose work is to review the report and provide the appropriate answers from the context.
User input will have the context required by you to answer user questions.
This context will begin with the token: ###Context.
The context contains references to specific portions of a document relevant to the user query.

User questions will begin with the token: ###Question.

Please answer only using the context provided in the input. Do not mention anything about the context in your final answer.

If the answer is not found in the context, respond "I don't know".
""".strip()

QNA_USER_MESSAGE_TEMPLATE = """
###Context
Here are some documents that are relevant to the question mentioned below.
{context}

###Question
{question}
""".strip()

GROUNDEDNESS_RATER_SYSTEM_MESSAGE = """
You are tasked with rating AI generated answers to questions posed by users.
You will be presented a question, context used by the AI system to generate the answer and an AI generated answer to the question.
In the input, the question will begin with ###Question, the context will begin with ###Context while the AI generated answer will begin with ###Answer.

Evaluation criteria:
The task is to judge the extent to which the metric is followed by the answer.
1 - The metric is not followed at all
2 - The metric is followed only to a limited extent
3 - The metric is followed to a good extent
4 - The metric is followed mostly
5 - The metric is followed completely

Metric:
The answer should be derived only from the information presented in the context

Instructions:
1. First write down the steps that are needed to evaluate the answer as per the metric.
2. Give a step-by-step explanation if the answer adheres to the metric considering the question and context as the input.
3. Next, evaluate the extent to which the metric is followed.
4. Use the previous information to rate the answer using the evaluaton criteria and assign a score.
""".strip()

RELEVANCE_RATER_SYSTEM_MESSAGE = """
You are tasked with rating AI generated answers to questions posed by users.
You will be presented a question, context used by the AI system to generate the answer and an AI generated answer to the question.
In the input, the question will begin with ###Question, the context will begin with ###Context while the AI generated answer will begin with ###Answer.

Evaluation criteria:
The task is to judge the extent to which the metric is followed by the answer.
1 - The metric is not followed at all
2 - The metric is followed only to a limited extent
3 - The metric is followed to a good extent
4 - The metric is followed mostly
5 - The metric is followed completely

Metric:
Relevance measures how well the answer addresses the main aspects of the question, based on the context.
Consider whether all and only the important aspects are contained in the answer when evaluating relevance.

Instructions:
1. First write down the steps that are needed to evaluate the context as per the metric.
2. Give a step-by-step explanation if the context adheres to the metric considering the question as the input.
3. Next, evaluate the extent to which the metric is followed.
4. Use the previous information to rate the context using the evaluaton criteria and assign a score.
""".strip()

EVAL_USER_MESSAGE_TEMPLATE = """
###Question
{question}

###Context
{context}

###Answer
{answer}
""".strip()

MEDICAL_DISCLAIMER = (
    "Educational prototype — not medical advice. Always consult a qualified "
    "clinician for diagnosis and treatment decisions."
)


def build_qa_prompt(context: str, question: str) -> str:
    """Assemble the grounded Q&A prompt from retrieved context + question."""
    user_message = QNA_USER_MESSAGE_TEMPLATE.format(context=context, question=question)
    return f"{QNA_SYSTEM_MESSAGE}\n{user_message}"


def build_eval_prompt(system_message: str, question: str, context: str, answer: str) -> str:
    """Assemble an LLM-as-judge prompt for groundedness or relevance."""
    user_message = EVAL_USER_MESSAGE_TEMPLATE.format(
        question=question, context=context, answer=answer
    )
    return f"[INST]{system_message}\n\n{{'user'}}: {user_message}\n[/INST]"
