# evaluate_rag.py
import os
import json
import asyncio
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app.ingest import load_and_chunk_docs
from app.retriever import HybridRAGRetriever
from app.generator import RAGGenerator

load_dotenv()

# ── Test questions with ground truth answers ──────────────────────────────────
# These are hand-crafted Q&A pairs based on what your corpus should contain
TEST_CASES = [
    {
        "question": "What is the role of TNF alpha in myocardial infarction?",
        "ground_truth": "TNF alpha is a pro-inflammatory cytokine that plays a significant role in myocardial infarction by promoting inflammation, contributing to cardiac tissue damage, and influencing the inflammatory response following ischemic injury."
    },
    {
        "question": "How do nanoparticles cross the blood-brain barrier?",
        "ground_truth": "Nanoparticles cross the blood-brain barrier through mechanisms including receptor-mediated transcytosis, adsorptive transcytosis, and passive diffusion. Surface functionalization with targeting ligands such as transferrin or ApoE enhances BBB penetration."
    },
    {
        "question": "What are the mechanisms of chemotherapy resistance in cancer cells?",
        "ground_truth": "Chemotherapy resistance in cancer cells develops through multiple mechanisms including drug efflux pump overexpression, DNA repair pathway upregulation, apoptosis evasion, and tumor microenvironment alterations."
    },
    {
        "question": "What is the role of microglia in neuroinflammation?",
        "ground_truth": "Microglia are the resident immune cells of the central nervous system that become activated during neuroinflammation, polarizing into pro-inflammatory M1 or anti-inflammatory M2 phenotypes and releasing cytokines that influence neuronal survival."
    },
    {
        "question": "How does mitochondrial dysfunction contribute to Alzheimer's disease?",
        "ground_truth": "Mitochondrial dysfunction in Alzheimer's disease leads to impaired energy metabolism, increased oxidative stress, and elevated ROS production, contributing to neuronal damage independent of amyloid and tau pathology."
    },
    {
        "question": "What biomarkers are used to track neurodegeneration in Alzheimer's disease?",
        "ground_truth": "Neurofilament light chain (NFL), GFAP, neurogranin, and SNAP-25 are biomarkers used to track neurodegeneration, synaptic loss, and neuroinflammation in Alzheimer's disease."
    },
    {
        "question": "What is the role of SGLT2 inhibitors in heart failure?",
        "ground_truth": "SGLT2 inhibitors reduce cardiovascular risk and improve outcomes in heart failure patients by reducing glucose reabsorption, promoting natriuresis, and potentially having direct cardiac protective effects."
    },
    {
        "question": "How does oxidative stress contribute to neuronal death?",
        "ground_truth": "Oxidative stress contributes to neuronal death through accumulation of reactive oxygen species that damage DNA, proteins, and lipids, leading to mitochondrial dysfunction and activation of apoptotic pathways."
    },
]

# ── Run pipeline for each question ───────────────────────────────────────────
def run_evaluation():
    print("Loading chunks and initializing pipeline...")
    chunks = load_and_chunk_docs("pmc_cardiology_oncology.json")
    retriever = HybridRAGRetriever(chunks)
    generator = RAGGenerator()

    questions     = []
    answers       = []
    contexts      = []
    ground_truths = []

    print(f"\nRunning {len(TEST_CASES)} test cases...\n")

    for i, tc in enumerate(TEST_CASES):
        q  = tc["question"]
        gt = tc["ground_truth"]
        print(f"[{i+1}/{len(TEST_CASES)}] {q[:60]}...")

        # retrieve
        docs = retriever.get_relevant_documents(q)

        # generate
        answer = generator.generate_answer(q, docs, history=[])

        # collect contexts as list of strings (RAGAS format)
        context_texts = [doc.page_content for doc in docs]

        questions.append(q)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(gt)

        print(f"  Answer: {answer[:100]}...")
        print(f"  Contexts retrieved: {len(context_texts)}\n")

    # ── Build RAGAS dataset ───────────────────────────────────────────────────
    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths
    })

    # ── Configure RAGAS to use your existing models ───────────────────────────
    # so you don't need an OpenAI key
    llm = LangchainLLMWrapper(ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    ))

    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ))

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("Running RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=emb,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("RAGAS EVALUATION RESULTS")
    print("="*50)

    scores = results.to_pandas()
    print(scores[["question", "faithfulness", "answer_relevancy",
                  "context_precision", "context_recall"]].to_string(index=False))

    print("\n── Averages ──")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        avg = scores[metric].mean()
        print(f"  {metric:<22} {avg:.3f}")

    # ── Save results ──────────────────────────────────────────────────────────
    scores.to_csv("ragas_results.csv", index=False)

    avg_scores = {
        metric: round(float(scores[metric].mean()), 3)
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    }

    with open("ragas_summary.json", "w") as f:
        json.dump(avg_scores, f, indent=2)

    print("\nSaved: ragas_results.csv and ragas_summary.json")
    print("\nPaste these into your README:")
    print(f"| Faithfulness      | {avg_scores['faithfulness']} |")
    print(f"| Answer Relevancy  | {avg_scores['answer_relevancy']} |")
    print(f"| Context Precision | {avg_scores['context_precision']} |")
    print(f"| Context Recall    | {avg_scores['context_recall']} |")

if __name__ == "__main__":
    run_evaluation()