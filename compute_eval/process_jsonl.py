import os
from typing import Dict, List
from compute_eval.data import write_jsonl


def correct_grammar(text: str) -> str:
    """
    Placeholder for grammar correction.
    Replace this with a call to a grammar correction model or API.
    """
    # For demonstration, just return the original text.
    # Integrate with a grammar correction tool for real use.
    return text


def analyze_struggles(transcript: str) -> List[Dict]:
    """
    Analyze transcript for struggles and suggest improvements.
    Returns a list of dicts with 'question', 'struggle', and 'improved_answer'.
    """
    struggles = []

    # Example struggle 1: Nervousness and filler words
    struggles.append(
        {
            "question": "Tell me about yourself / your background.",
            "struggle": "Frequent use of filler words, run-on sentences, and expressing nervousness.",
            "improved_answer": (
                "Prepare a concise summary of your background. "
                "Speak slowly and confidently, focusing on key achievements and transitions. "
                "Avoid filler words and unnecessary details."
            ),
        }
    )

    # Example struggle 2: Overly technical or lengthy answers
    struggles.append(
        {
            "question": "Describe a technical project or challenge.",
            "struggle": "Over-explaining technical details without summarizing business impact.",
            "improved_answer": (
                "Use the STAR (Situation, Task, Action, Result) format. "
                "Briefly describe the context, your role, the actions you took, and the outcome. "
                "Highlight the business value or impact."
            ),
        }
    )

    # Example struggle 3: Motivation for role change
    struggles.append(
        {
            "question": "Why do you want to move into a business development role?",
            "struggle": "Long explanation, lack of direct answer.",
            "improved_answer": (
                "Clearly state your motivation for the transition, relevant skills, and how you will add value. "
                "Be direct and relate your answer to the company’s needs."
            ),
        }
    )

    # Example struggle 4: Handling multiple stakeholders
    struggles.append(
        {
            "question": "How do you handle competing priorities or stakeholders?",
            "struggle": "Describing consensus-building but lacking a concise summary.",
            "improved_answer": (
                "Summarize your approach to stakeholder management, then provide a specific example. "
                "Emphasize communication, alignment, and delivering results."
            ),
        }
    )

    return struggles


def process_transcript(transcript: str, output_file: str):
    """
    Process the transcript: correct grammar and analyze struggles.
    """
    corrected = correct_grammar(transcript)
    struggles = analyze_struggles(transcript)

    output = {
        "original_transcript": transcript,
        "corrected_transcript": corrected,
        "analysis": {"struggles": struggles},
    }

    write_jsonl(output_file, [output])


# Remove hardcoded file references and provide CLI usage or function only
if __name__ == "__main__":
    print(
        "This script provides transcript processing functions. Import and use process_transcript() in your workflow."
    )
