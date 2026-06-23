# evaluate_rag.py
import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.ingest import load_and_chunk_docs
from app.retriever import HybridRAGRetriever
from app.generator import RAGGenerator

load_dotenv()

TEST_CASES = [
    {
        "question": "What is the role of TNF-α in myocardial infarction?",
        "ground_truth": "TNF-α is a pro-inflammatory cytokine that promotes inflammation and contributes to cardiac tissue damage following myocardial infarction."
    },
    {
        "question": "How do nanoparticles cross the blood-brain barrier?",
        "ground_truth": "Nanoparticles cross the blood-brain barrier through receptor-mediated transcytosis, adsorptive transcytosis, and surface functionalization with targeting ligands such as transferrin or ApoE."
    },
    {
        "question": "What are the mechanisms of chemotherapy resistance in cancer cells?",
        "ground_truth": "Chemotherapy resistance develops through drug efflux pump overexpression, DNA repair pathway upregulation, apoptosis evasion, and tumor microenvironment alterations."
    },
    {
        "question": "What is the role of microglia in neuroinflammation?",
        "ground_truth": "Microglia are resident immune cells of the CNS that polarize into pro-inflammatory M1 or anti-inflammatory M2 phenotypes and release cytokines that influence neuronal survival."
    },
    {
        "question": "How does mitochondrial dysfunction contribute to Alzheimer's disease?",
        "ground_truth": "Mitochondrial dysfunction leads to impaired energy metabolism, increased oxidative stress, and elevated ROS production, contributing to neuronal damage independent of amyloid and tau pathology."
    },
    {
        "question": "What biomarkers track neurodegeneration in Alzheimer's disease?",
        "ground_truth": "Neurofilament light chain, GFAP, neurogranin, and SNAP-25 are biomarkers used to track neurodegeneration and synaptic loss in Alzheimer's disease."
    },
    {
        "question": "What is the role of SGLT2 inhibitors in heart failure?",
        "ground_truth": "SGLT2 inhibitors reduce cardiovascular risk in heart failure by reducing glucose reabsorption, promoting natriuresis, and providing direct cardioprotective effects."
    },
    {
        "question": "How does oxidative stress contribute to neuronal death?",
        "ground_truth": "Oxidative stress causes neuronal death through reactive oxygen species that damage DNA, proteins, and lipids, activating mitochondrial dysfunction and apoptotic pathways."
    },
]

# ── Metric 1: Faithfulness ────────────────────────────────────────────────────
# Does the answer only contain claims supported by the context?
def score_faithfulness(answer: str, contexts: list[str], llm) -> float:
    context_combined = "\n\n".join(contexts[:5])
    prompt = f"""You are evaluating whether an AI answer is faithful to the source context.

Context:
{context_combined}

Answer:
{answer}

Score how faithful the answer is to the context on a scale of 0.0 to 1.0:
- 1.0: Every claim in the answer is directly supported by the context
- 0.5: Most claims are supported but some are inferred or added
- 0.0: Answer contains claims not found in the context at all

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

    try:
        result = llm.invoke(prompt).content.strip()
        return float(result)
    except Exception:
        return 0.0


# ── Metric 2: Answer Relevancy ────────────────────────────────────────────────
# Is the answer actually relevant to the question? (embedding similarity)
def score_answer_relevancy(question: str, answer: str, embedder) -> float:
    try:
        q_emb = embedder.embed_query(question)
        a_emb = embedder.embed_query(answer)
        sim = cosine_similarity([q_emb], [a_emb])[0][0]
        return round(float(sim), 3)
    except Exception:
        return 0.0


# ── Metric 3: Context Recall ──────────────────────────────────────────────────
# Did the retrieved context contain the information needed to answer?
def score_context_recall(ground_truth: str, contexts: list[str], llm) -> float:
    context_combined = "\n\n".join(contexts[:3])
    prompt = f"""You are evaluating whether retrieved context contains enough 
information to support a ground truth answer.

Ground Truth Answer:
{ground_truth}

Retrieved Context:
{context_combined}

Score how well the context covers the ground truth on a scale of 0.0 to 1.0:
- 1.0: Context contains all information needed to produce the ground truth
- 0.5: Context contains some relevant information but is incomplete
- 0.0: Context is missing the information needed

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

    try:
        result = llm.invoke(prompt).content.strip()
        return float(result)
    except Exception:
        return 0.0


# ── Metric 4: Context Precision ───────────────────────────────────────────────
# Are the retrieved chunks actually useful for the question?
def score_context_precision(question: str, contexts: list[str], embedder) -> float:
    try:
        q_emb = embedder.embed_query(question)
        scores = []
        for ctx in contexts:
            c_emb = embedder.embed_query(ctx[:500])
            sim = cosine_similarity([q_emb], [c_emb])[0][0]
            scores.append(float(sim))
        return round(float(np.mean(scores)), 3) if scores else 0.0
    except Exception:
        return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
def run_evaluation():
    print("Loading pipeline...")
    chunks    = load_and_chunk_docs("pmc_cardiology_oncology.json")
    retriever = HybridRAGRetriever(chunks)
    generator = RAGGenerator()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    results = []

    print(f"\nEvaluating {len(TEST_CASES)} questions...\n")

    for i, tc in enumerate(TEST_CASES):
        q  = tc["question"]
        gt = tc["ground_truth"]
        print(f"[{i+1}/{len(TEST_CASES)}] {q[:65]}...")

        docs    = retriever.get_relevant_documents(q)
        answer  = generator.generate_answer(q, docs, history=[])
        contexts = [doc.page_content for doc in docs]

        f  = score_faithfulness(answer, contexts, llm)
        ar = score_answer_relevancy(q, answer, embedder)
        cr = score_context_recall(gt, contexts, llm)
        cp = score_context_precision(q, contexts, embedder)

        results.append({
            "question":          q,
            "answer":            answer,
            "faithfulness":      f,
            "answer_relevancy":  ar,
            "context_recall":    cr,
            "context_precision": cp,
        })

        print(f"  faithfulness={f:.2f}  answer_relevancy={ar:.2f}  context_recall={cr:.2f}  context_precision={cp:.2f}\n")

    # ── Averages ──────────────────────────────────────────────────────────────
    metrics = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    avg = {m: round(float(np.mean([r[m] for r in results])), 3) for m in metrics}

    print("=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)
    for m, v in avg.items():
        bar = "█" * int(v * 20)
        print(f"  {m:<22} {v:.3f}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open("ragas_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("ragas_summary.json", "w") as f:
        json.dump(avg, f, indent=2)

    print("\nSaved: ragas_results.json + ragas_summary.json")

    print("\n── Paste into README ──")
    print("| Metric             | Score |")
    print("|--------------------|-------|")
    for m, v in avg.items():
        label = m.replace("_", " ").title()
        print(f"| {label:<19}| {v}   |")


if __name__ == "__main__":
    run_evaluation()