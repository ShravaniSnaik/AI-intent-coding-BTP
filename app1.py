import os
import sys
import io
from contextlib import contextmanager
import spacy
import torch
import numpy as np
import re
import json
import itertools
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from google import genai
from google.genai import types
from pydantic import BaseModel
import penman
import amrlib
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any

# =====================================================================
# UI CONFIGURATION
# =====================================================================
st.set_page_config(page_title="IntentCoding Agentic Pipeline", layout="wide")
st.title("🚀 IntentCoding: Neuro-Symbolic Agentic Pipeline")

# =====================================================================
# OUTPUT CAPTURE (Streams your print() statements to Streamlit)
# =====================================================================
@contextmanager
def capture_stdout():
    new_out = io.StringIO()
    old_out = sys.stdout
    try:
        sys.stdout = new_out
        yield new_out
    finally:
        sys.stdout = old_out

# =====================================================================
# MODEL CACHING (Prevents reloading on every UI click)
# =====================================================================
@st.cache_resource
def load_models():
    print("Loading NLP, NLI, SBERT & AMR Models...")
    nlp = spacy.load("en_core_web_sm")
    nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    try:
        amr_stog = amrlib.load_stog_model('./model_parse_xfm_bart_base-v0_1_0')
    except Exception as e:
        print(f"Warning: AMR Model failed to load. {e}")
        amr_stog = None
        
    return nlp, nli_tokenizer, nli_model, sbert_model, amr_stog

with st.spinner("Loading models into memory..."):
    with capture_stdout() as init_out:
        nlp, nli_tokenizer, nli_model, sbert_model, amr_stog = load_models()
    if init_out.getvalue().strip():
        st.code(init_out.getvalue(), language="text")

try:
    from google.colab import userdata
    api_key = userdata.get('GOOGLE_API_KEY2')
except ImportError:
    api_key = os.environ.get('GOOGLE_API_KEY2')

client = genai.Client(api_key=api_key)

# # Pydantic Schemas 
# class OrthogonalGap(BaseModel):
#     missing_parameter: str
#     options: list[str]
#     is_applicable: bool

# class GapsExtraction(BaseModel):
#     orthogonal_gaps: list[OrthogonalGap]

class OrthogonalGap(BaseModel):
    missing_parameter: str
    options: list[str]
    is_applicable: bool

# REPLACES GapsExtraction
class Phase2Extraction(BaseModel):
    core_features: list[str]
    orthogonal_gaps: list[OrthogonalGap]

# =====================================================================
# ZERO-SHOT BASELINE FUNCTION (For Panel Comparison)
# =====================================================================
def get_zero_shot_baseline(prompt: str) -> str:
    """Queries the LLM directly without the pipeline for a true naive benchmark."""
    sys_prompt = (
        "Convert the following informal system requirements into formal specifications."
    )
    response = client.models.generate_content(
        model='gemini-flash-lite-latest',
        contents=f"{sys_prompt}\n\nInformal Prompt: '{prompt}'",
        config=types.GenerateContentConfig(temperature=0.3)
    )
    return response.text

# =====================================================================
# PHASE 1: STEP 0 (EARS ENFORCED DEDUPLICATION)
# =====================================================================
def step_0_hybrid_seeding(prompt: str) -> list:
    print("\n" + "="*90)
    print("[PHASE 1: STEP 0] DYNAMIC SEEDING (Lexical Cleaner + Advanced SRL + Fallback Union)")
    print("="*90)
    print(f"[INPUT] Raw Informal Prompt:\n '{prompt}'\n")

    print("[PROCESS] Executing Lexical Cleaner (LLM) to resolve pronouns...")
    sys_prompt_cleaner = (
        "You are a Lexical Coreference Resolver. Your only job is to rewrite the provided informal prompt "
        "by replacing all vague pronouns (e.g., 'it', 'this', 'they', 'them') with the exact explicit nouns "
        "they refer to based on the context. Do not change the meaning, tone, or structure of the sentences. "
        "Just output the rewritten prompt."
    )
    response_cleaner = client.models.generate_content(
        model='gemini-flash-lite-latest',
        contents=f"{sys_prompt_cleaner}\n\nINFORMAL PROMPT: '{prompt}'",
        config=types.GenerateContentConfig(temperature=0.0)
    )
    cleaned_prompt = response_cleaner.text.strip()
    print(f"[OUTPUT] Lexical Cleaner Output:\n '{cleaned_prompt}'\n")

    print("[PROCESS] Executing Deterministic SRL (Agent -> Verb -> Target -> Negation)...")
    doc = nlp(cleaned_prompt)
    srl_set = []
    for token in doc:
        if token.pos_ == "VERB":
            agent = "system"
            target = ""
            negation = ""
            for child in token.children:
                if child.dep_ in ["nsubj", "agent"]: agent = child.lemma_
                if child.dep_ in ["dobj", "pobj", "nsubjpass"]: target = child.lemma_
                if child.dep_ == "neg": negation = "NOT "
            if target:
                constraint = f"The {agent} must {negation}{token.lemma_} the {target}."
                srl_set.append(constraint)

    print(f"[OUTPUT] Advanced SRL Extracted Constraints:\n {json.dumps(srl_set, indent=2)}\n")

    print("[PROCESS] Executing Neural Fallback (LLM) scanning for missing constraints...")
    sys_prompt = (
        "You are a Requirements Fallback Engine. I will provide an informal prompt and the "
        "rigid constraints already successfully extracted by our deterministic SRL model.\n"
        "Your task:\n"
        "1. Analyze the informal prompt for any architectural, temporal, or domain-specific constraints that the SRL missed.\n"
        "2. If the SRL missed anything, write the missing constraints as simple declarative sentences.\n"
        "3. If the SRL captured everything, return an empty list [].\n"
        "Return ONLY a valid JSON list of strings. Do not repeat constraints the SRL already found."
    )
    response = client.models.generate_content(
        model='gemini-flash-lite-latest',
        contents=f"{sys_prompt}\n\nINFORMAL PROMPT: '{cleaned_prompt}'\nSRL EXTRACTED: {srl_set}",
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
    )
    llm_fallback_set = json.loads(response.text)

    if llm_fallback_set:
        print(f"[OUTPUT] LLM Neural Fallback Discovered Constraints:\n {json.dumps(llm_fallback_set, indent=2)}\n")
    else:
        print(f"[OUTPUT] LLM Neural Fallback: SRL coverage 100%. No fallback required.\n")

    print("[PROCESS] Executing Semantic Deduplication (EARS CNL Enforcement)...")
    sys_prompt_dedup = (
        "I am providing two lists of system constraints extracted from the same prompt: an SRL list and an LLM Fallback list.\n"
        "Merge them into a single, comprehensive list of absolute ground truth constraints.\n"
        "CRITICAL RULES:\n"
        "1. RESOLVE CONTRADICTIONS: Delete broken SRL fragments if they contradict detailed Fallback constraints.\n"
        "2. REMOVE SUBSUMPTION: If one constraint is a vague subset of another, delete the vague one.\n"
        "3. STRICT EARS FORMAT: You MUST format every final constraint using the EARS Controlled Natural Language. "
        "If there is a condition, start with it (e.g., 'While the rotors are spinning, the system must...'). "
        "If there is no condition, use the ubiquitous template (e.g., 'The memory must encrypt data.').\n"
        "Return a valid JSON list of strings."
    )
    res_dedup = client.models.generate_content(
        model='gemini-flash-lite-latest',
        contents=f"{sys_prompt_dedup}\n\nList 1 (SRL): {srl_set}\nList 2 (Fallback): {llm_fallback_set}",
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
    )
    final_ground_truth = json.loads(res_dedup.text)
    print(f"[FINAL OUTPUT: STEP 0] Ground Truth Constraints:\n {json.dumps(final_ground_truth, indent=2)}\n")

    return final_ground_truth

# =====================================================================
# GATES (NLI & AMR)
# =====================================================================
def step_2_dual_nli_matrix(prompt: str, feature: str, quiet=False):
    premise = f"Context: {prompt}"
    hypo_pos = feature
    feature_lower = feature[0].lower() + feature[1:] if feature else ""
    hypo_neg = f"It is false that {feature_lower}"

    if not quiet:
        print(f"      [INPUT] NLI Premise : '{premise}'")
        print(f"      [INPUT] NLI Hypo (+) : '{hypo_pos}'")
        print(f"      [INPUT] NLI Hypo (-) : '{hypo_neg}'")

    inputs_pos = nli_tokenizer(premise, hypo_pos, return_tensors="pt")
    inputs_neg = nli_tokenizer(premise, hypo_neg, return_tensors="pt")

    with torch.no_grad():
        p_pos_entail = torch.softmax(nli_model(**inputs_pos).logits, dim=1).squeeze()[2].item()
        p_neg_entail = torch.softmax(nli_model(**inputs_neg).logits, dim=1).squeeze()[2].item()

    if p_pos_entail > 0.40 and p_neg_entail < 0.40: return "ENTAILED", p_pos_entail, p_neg_entail
    elif p_neg_entail > 0.40 and p_pos_entail < 0.40: return "CONTRADICTION", p_pos_entail, p_neg_entail
    else: return "AMBIGUOUS", p_pos_entail, p_neg_entail

def step_3_amr_relevance_gate(prompt: str, feature: str, mode: str) -> bool:
    print(f"      [PROCESS] Generating AMR Topological Graphs for Premise and Feature (Mode: {mode.upper()})...")
    try:
        prompt_amr = amr_stog.parse_sents([prompt])[0]
        feature_amr = amr_stog.parse_sents([feature])[0]
        prompt_graph = penman.decode(prompt_amr)
        feature_graph = penman.decode(feature_amr)

        ignore_words = {'system', 'user', 'data', 'database', 'application', 'software', 'information', 'it', 'record'}
        prompt_concepts = {t.target for t in prompt_graph.instances() if isinstance(t.target, str) and len(t.target) > 2} - ignore_words
        feature_concepts = {t.target for t in feature_graph.instances() if isinstance(t.target, str) and len(t.target) > 2} - ignore_words

        overlap = prompt_concepts.intersection(feature_concepts)
        
        # Calculate proportional overlap relative to the generated feature
        overlap_ratio = len(overlap) / len(feature_concepts) if len(feature_concepts) > 0 else 0.0

        print(f"      [OUTPUT] AMR Overlap Ratio: {overlap_ratio:.2f} ({len(overlap)}/{len(feature_concepts)} nodes)")

        if mode == "core":
            # STRICT: A core feature should mostly consist of concepts already in the prompt.
            # We require at least 50% of the core feature's concepts to exist in the prompt.
            passed = overlap_ratio >= 0.40
        elif mode == "gap":
            # LENIENT: A gap introduces novel concepts but must share at least ONE anchor entity 
            # (like the subject or object) with the original prompt.
            passed = len(overlap) > 0
        else:
            passed = False

        if passed:
            print("      [OUTPUT] ✅ AMR Gate Passed")
        else:
            print("      [OUTPUT] 🚫 AMR Gate Failed")
            
        return passed

    except Exception as e:
        print(f"      [WARNING] AMR Parsing failed: {e}")
        return False
# =====================================================================
# PHASE 3: DETERMINISTIC BOOLEAN COMPILER (Zero-LLM)
# =====================================================================
def compile_to_boolean(sentence: str) -> str:
    doc = nlp(sentence)
    root = next((token for token in doc if token.dep_ == "ROOT"), None)
    if not root: return sentence
    
    advcl = next((child for child in root.children if child.dep_ == "advcl"), None)
    lhs_tokens, rhs_tokens = [], []
    
    drop_words = {"must", "shall", "will", "can", "should"}
    
    for token in doc:
        if token.is_punct or token.is_space: continue
        if token.pos_ in ["DET", "AUX"] or token.text.lower() in drop_words: continue
        
        if advcl and token in advcl.subtree:
            if token.dep_ != "mark":
                lhs_tokens.append(token.lemma_.lower())
        else:
            if token.dep_ == "neg": rhs_tokens.append("NOT")
            else: rhs_tokens.append(token.lemma_.lower())
                
    lhs_var = "_".join(lhs_tokens) if lhs_tokens else "TRUE"
    rhs_var = "_".join(rhs_tokens)
    
    lhs_var = re.sub(r'_+', '_', lhs_var).strip('_')
    rhs_var = re.sub(r'_+', '_', rhs_var).strip('_')
    
    # NEW: Aggressive scrubbing of starting markers
    markers = ["if_", "when_", "whenever_", "while_"]
    for marker in markers:
        if lhs_var.startswith(marker):
            lhs_var = lhs_var[len(marker):]
        if rhs_var.startswith(marker):
            rhs_var = rhs_var[len(marker):]
            
    if lhs_var != "TRUE" and lhs_var:
        return f"({lhs_var}) -> ({rhs_var})"
    else:
        return f"({rhs_var})"
# =====================================================================
# END-TO-END METRICS EVALUATION (PER UNIVERSE)
# =====================================================================
def evaluate_universe(raw_prompt: str, core_rules: list, gap_rules: list):
    raw_sentences = [s.strip() for s in re.split(r'[.!?\n]', raw_prompt) if len(s.strip()) > 5]
    universe_text = " ".join(core_rules + gap_rules)
    universe_embedding = sbert_model.encode([universe_text], convert_to_tensor=True)
    
    covered = 0
    if raw_sentences:
        raw_embeddings = sbert_model.encode(raw_sentences, convert_to_tensor=True)
        cosine_scores = util.cos_sim(raw_embeddings, universe_embedding)
        for i in range(len(raw_sentences)):
            if cosine_scores[i][0].item() > 0.45:
                covered += 1
        s_icmw = covered / len(raw_sentences)
    else:
        s_icmw = 1.0

    f_scores = []
    for gap in gap_rules:
        _, _, p_neg = step_2_dual_nli_matrix(raw_prompt, gap, quiet=True)
        consistency_score = 1.0 - p_neg 
        f_scores.append(consistency_score)
    f_icmw = np.mean(f_scores) if f_scores else 1.0

    rel_scores = []
    try:
        prompt_amr = amr_stog.parse_sents([raw_prompt])[0]
        prompt_graph = penman.decode(prompt_amr)
        ignore_words = {'system', 'user', 'data', 'database', 'application', 'software', 'information', 'it', 'record'}
        prompt_concepts = {t.target for t in prompt_graph.instances() if isinstance(t.target, str) and len(t.target) > 2} - ignore_words
        
        for gap in gap_rules:
            try:
                gap_amr = amr_stog.parse_sents([gap])[0]
                gap_graph = penman.decode(gap_amr)
                gap_concepts = {t.target for t in gap_graph.instances() if isinstance(t.target, str) and len(t.target) > 2} - ignore_words
                
                intersection = prompt_concepts.intersection(gap_concepts)
                smallest_set_size = min(len(prompt_concepts), len(gap_concepts))
                rel = len(intersection) / smallest_set_size if smallest_set_size > 0 else 0.0
                rel_scores.append(rel)
            except:
                rel_scores.append(0.0)
        relevance = np.mean(rel_scores) if rel_scores else 1.0
    except:
        relevance = 0.0 

    return s_icmw, f_icmw, relevance

# =====================================================================
# LANGGRAPH AGENTIC STATE & NODES 
# =====================================================================
class PipelineState(TypedDict):
    prompt: str
    ground_truth: List[str]
    verified_core_rules: List[str]  # NEW: Stores gates-verified core rules
    core_text: str
    iteration: int
    max_iterations: int
    feedback: str
    errors_to_feedback: List[str]
    extracted_gaps: Any 
    validated_gaps: Dict[str, List[str]]
    combinations: List[Any]

def extractor_node(state: PipelineState):
    gt = step_0_hybrid_seeding(state["prompt"])
    print(f"\n{'='*90}")
    print(f"🚀 RUNNING PHASE 2: PURE GAP-GENERATION LOOP")
    print(f"{'='*90}")
    return {
        "ground_truth": gt, 
        "verified_core_rules": [], # Initialize empty
        "core_text": " ".join(gt), 
        "iteration": 0, 
        "feedback": "", 
        "validated_gaps": {}, 
        "errors_to_feedback": []
    }
def generator_node(state: PipelineState):
    iteration = state["iteration"] + 1
    print(f"\n{'='*40} ITERATION {iteration} {'='*40}")
    print(f"\n[PROCESS] Phase 2 Generator synthesizing Core Features & Gaps from Phase 1 hints...")
    
    sys_prompt = (
        "You are a Software Architect. You have two tasks based on the Informal Prompt and the Phase 1 Ground Truth Hints:\n"
        "1. `core_features`: Formalize the provided 'GROUND TRUTH HINTS' into complete, non-contradictory core requirement sentences using strict EARS (Easy Approach to Requirements Syntax) format.\n"
        "2. `orthogonal_gaps`: Identify missing parameters (edge cases not covered by core features) and provide mutually exclusive options.\n"
        "CRITICAL: Every single option generated inside `orthogonal_gaps` MUST be a complete, standalone declarative sentence using a valid EARS template. \n"
        "Allowed templates:\n"
        " - Ubiquitous: 'The [system] must [action].'\n"
        " - Event-Driven: 'When [trigger], the [system] must [action].'\n"
        " - State-Driven: 'While [state], the [system] must [action].'\n"
        " - Unwanted Behavior: 'If [hazard], the [system] must [action].'\n"
        "NEVER output noun phrases or fragments.\n\n"
        f"GROUND TRUTH HINTS (Topics that MUST be formalized into core_features): {state['ground_truth']}\n"
        f"{state['feedback']}"
    )
    
    response = client.models.generate_content(
        model='gemini-flash-lite-latest',
        contents=f"{sys_prompt}\n\nUser Prompt: '{state['prompt']}'",
        config=types.GenerateContentConfig(
            response_mime_type="application/json", 
            response_schema=Phase2Extraction, 
            temperature=0.9
        )
    )
    
    # Parse both core features and orthogonal gaps from the LLM
    extraction = Phase2Extraction.model_validate_json(response.text)

    print(f"[OUTPUT] LLM Generated {len(extraction.core_features)} Core Features and {len(extraction.orthogonal_gaps)} Gaps.")
    return {"extracted_gaps": extraction, "iteration": iteration}

def validator_node(state: PipelineState):
    extraction = state["extracted_gaps"] # This now contains both core_features and orthogonal_gaps
    raw_prompt = state["prompt"] 
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]
    errors_to_feedback = []
    
    print(f"\n[PROCESS] Routing Phase 2 Generated Core Features & Gaps -> NLI decision -> AMR Gate")
    
    # ---------------------------------------------------------
    # LOOP A: VALIDATE PHASE 2 CORE FEATURES 
    # ---------------------------------------------------------
    verified_core_rules = []
    for core_rule in extraction.core_features: 
        print(f"\n ↳ [INPUT] Evaluating Phase 2 Core Feature: [{core_rule}]")
        result, p_pos, p_neg = step_2_dual_nli_matrix(raw_prompt, core_rule)
        
        if result == "ENTAILED":
            print("      [OUTPUT] ✅ ENTAILMENT -> Verified Core Feature Bin")
            verified_core_rules.append(core_rule)
        elif result == "CONTRADICTION":
            print("      [OUTPUT] ❌ CONTRADICTION -> Feedback Bin")
            err = f"[PATH]: Generator -> Core Feature -> Contradict -> Feedback Bin\nSYMPTOM: Core rule '{core_rule}' contradicts the informal prompt."
            errors_to_feedback.append(err)
        elif result == "AMBIGUOUS":
            print("      [OUTPUT] ⚠️ AMBIGUOUS -> Proceeding to AMR Gate") # <--- Added print statement here
            if step_3_amr_relevance_gate(raw_prompt, core_rule, mode="core"):
                print("      [OUTPUT] ✅ AMR Gate Passed -> Verified Core Feature Bin")
                verified_core_rules.append(core_rule)
            else:
                print("      [OUTPUT] 🚫 AMR Gate Failed -> Feedback Bin")
                err = f"[PATH]: Generator -> Core Feature -> Ambiguous -> AMR gate fail -> Feedback Bin\nSYMPTOM: Core rule '{core_rule}' failed AMR relevance against the informal prompt."
                errors_to_feedback.append(err)

    # ---------------------------------------------------------
    # LOOP B: VALIDATE PHASE 2 GAP FEATURES 
    # ---------------------------------------------------------
    validated_gaps = {}
    for gap in extraction.orthogonal_gaps:
        if not gap.is_applicable: continue
        print(f"\n ↳ [INPUT] Evaluating Gap Dimension: [{gap.missing_parameter}]")
        valid_options = []
        for option in gap.options:
            result, p_pos, p_neg = step_2_dual_nli_matrix(raw_prompt, option)

            if result == "ENTAILED":
                print("      [OUTPUT] ✅ ENTAILMENT -> Validated Gaps Bin")
                valid_options.append(option)
            elif result == "CONTRADICTION":
                print("      [OUTPUT] ❌ CONTRADICTION -> Feedback Bin")
                err = f"[PATH]: Generator -> Orthogonal Gap -> Contradict -> Feedback Bin\nSYMPTOM: Option '{option}' contradicts the informal prompt."
                errors_to_feedback.append(err)
            elif result == "AMBIGUOUS":
                print("      [OUTPUT] ⚠️ AMBIGUOUS -> Proceeding to AMR Gate") # <--- Added print statement here
                if step_3_amr_relevance_gate(raw_prompt, option, mode="gap"):
                    print("      [OUTPUT] ✅ AMR Gate Passed -> Validated Gaps Bin")
                    valid_options.append(option)
                else:
                    print("      [OUTPUT] 🚫 AMR Gate Failed -> Feedback Bin")
                    err = f"[PATH]: Generator -> Orthogonal Gap -> Ambiguous -> AMR gate fail -> Feedback Bin\nSYMPTOM: Option '{option}' failed AMR relevance against the informal prompt."
                    errors_to_feedback.append(err)

        if valid_options: validated_gaps[gap.missing_parameter] = valid_options

    if not errors_to_feedback:
        print(f"\n🏆 CONVERGENCE REACHED AT ITERATION {iteration}!")
        feedback = ""
    else:
        feedback = "CRITICAL ARCHITECTURAL ERRORS TO FIX:\n" + "\n".join(errors_to_feedback)

    return {
        "errors_to_feedback": errors_to_feedback, 
        "feedback": feedback, 
        "verified_core_rules": verified_core_rules, 
        "validated_gaps": validated_gaps
    }
    # ---------------------------------------------------------
    # LOOP B: VALIDATE PHASE 2 GAP FEATURES 
    # ---------------------------------------------------------
    validated_gaps = {}
    for gap in extraction.orthogonal_gaps:
        if not gap.is_applicable: continue
        print(f"\n ↳ [INPUT] Evaluating Gap Dimension: [{gap.missing_parameter}]")
        valid_options = []
        for option in gap.options:
            result, p_pos, p_neg = step_2_dual_nli_matrix(raw_prompt, option)

            if result == "ENTAILED":
                print("      [OUTPUT] ✅ ENTAILMENT -> Validated Gaps Bin")
                valid_options.append(option)
            elif result == "CONTRADICTION":
                print("      [OUTPUT] ❌ CONTRADICTION -> Feedback Bin")
                err = f"[PATH]: Generator -> Orthogonal Gap -> Contradict -> Feedback Bin\nSYMPTOM: Option '{option}' contradicts the informal prompt."
                errors_to_feedback.append(err)
            elif result == "AMBIGUOUS":
                if step_3_amr_relevance_gate(raw_prompt, option):
                    print("      [OUTPUT] ✅ AMR Gate Passed -> Validated Gaps Bin")
                    valid_options.append(option)
                else:
                    print("      [OUTPUT] 🚫 AMR Gate Failed -> Feedback Bin")
                    err = f"[PATH]: Generator -> Orthogonal Gap -> Ambiguous -> AMR gate fail -> Feedback Bin\nSYMPTOM: Option '{option}' failed AMR relevance against the informal prompt."
                    errors_to_feedback.append(err)

        if valid_options: validated_gaps[gap.missing_parameter] = valid_options

    if not errors_to_feedback:
        print(f"\n🏆 CONVERGENCE REACHED AT ITERATION {iteration}!")
        feedback = ""
    else:
        feedback = "CRITICAL ARCHITECTURAL ERRORS TO FIX:\n" + "\n".join(errors_to_feedback)

    return {
        "errors_to_feedback": errors_to_feedback, 
        "feedback": feedback, 
        "verified_core_rules": verified_core_rules, 
        "validated_gaps": validated_gaps
    }
def router(state: PipelineState):
    if not state["errors_to_feedback"]: return "Assembly"
    if state["iteration"] >= state["max_iterations"]: return "Assembly"
    return "Generator"

def assembly_node(state: PipelineState):
    print(f"\n{'='*90}")
    print(f"🏆 PHASE 3: FINAL UNIVERSES, BOOLEAN COMPILATION & EVALUATION")
    print(f"{'='*90}")

    gap_lists = list(state["validated_gaps"].values())
    combinations = list(itertools.product(*gap_lists)) if gap_lists else [()]
    
    # We now strictly assemble using the Verified Core Rules
    verified_core = state["verified_core_rules"]

    for i, combo in enumerate(combinations):
        print(f"\n🪐 UNIVERSE {i+1}:")
        
        print("   [ENGLISH FORMULATION]")
        for gt in verified_core: print(f"    - [CORE] {gt}")
        for gap_feature in combo: print(f"    - [GAP]  {gap_feature}")
        
        print("\n   [BOOLEAN LOGIC FORMULATION]")
        for gt in verified_core: 
            print(f"    - [CORE_LOGIC] {compile_to_boolean(gt)}")
        for gap_feature in combo: 
            print(f"    - [GAP_LOGIC]  {compile_to_boolean(gap_feature)}")
            
        print("\n   [END-TO-END METRICS]")
        # Metrics use the raw prompt as the ultimate truth anchor
        s_icmw, f_icmw, relevance = evaluate_universe(state["prompt"], verified_core, list(combo))
        print(f"    - Completeness (s-ICMw) : {s_icmw:.2f}")
        print(f"    - Consistency (f-ICMw)  : {f_icmw:.2f}")
        print(f"    - Relevance (AMR Index) : {relevance:.2f}")

    print(f"\n{'-'*90}\n🏁 PIPELINE COMPLETE.")
    return {"combinations": combinations}
# =====================================================================
# BUILD LANGGRAPH
# =====================================================================
workflow = StateGraph(PipelineState)
workflow.add_node("Extractor", extractor_node)
workflow.add_node("Generator", generator_node)
workflow.add_node("Validator", validator_node)
workflow.add_node("Assembly", assembly_node)

workflow.set_entry_point("Extractor")
workflow.add_edge("Extractor", "Generator")
workflow.add_edge("Generator", "Validator")
workflow.add_conditional_edges("Validator", router, {"Assembly": "Assembly", "Generator": "Generator"})
workflow.add_edge("Assembly", END)

app_graph = workflow.compile()

# =====================================================================
# STREAMLIT UI EXECUTION
# =====================================================================
st.sidebar.header("Architecture Visualization")
try:
    st.sidebar.image(app_graph.get_graph().draw_mermaid_png(), caption="LangGraph Topology")
except Exception:
    st.sidebar.text("Mermaid graph rendering requires internet access.")

st.markdown("### Enter Informal Specification Prompt:")
default_prompt = (
    "The bus arbiter must ensure that no two master devices are granted access to the system bus at the same time. "
    "Whenever a high-priority master requests the bus, the system must grant it access immediately. "
    "Furthermore, a master device must release the bus lock after four clock cycles."
)
user_prompt = st.text_area("Prompt", default_prompt, height=100)

if st.button("Execute & Compare Baseline"):
    st.divider()
    
    # ---------------------------------------------------------
    # 1. TOP SECTION: ZERO-SHOT BASELINE
    # ---------------------------------------------------------
    st.markdown("### 🤖 Standard Zero-Shot LLM (Baseline)")
    with st.spinner("Generating raw LLM response..."):
        baseline_result = get_zero_shot_baseline(user_prompt)
        st.success("Generation Complete (No Validation)")
        st.info(baseline_result)
        
    st.divider() # Visual separator between the two outputs
        
    # ---------------------------------------------------------
    # 2. BOTTOM SECTION: AGENTIC PIPELINE
    # ---------------------------------------------------------
    st.markdown("### 🚀 Neuro-Symbolic Agentic Pipeline")
    trace_container = st.container()
    
    with st.spinner("Pipeline Running... Standby for metrics."):
        with capture_stdout() as out:
            for event in app_graph.stream({"prompt": user_prompt, "max_iterations": 15}):
                for node_name, state_update in event.items():
                    printed_text = out.getvalue()
                    out.truncate(0)
                    out.seek(0)
                    
                    with trace_container.expander(f"⚙️ Node Executed: {node_name}", expanded=True):
                        if printed_text.strip():
                            st.code(printed_text, language="text")
    st.success("Agentic Pipeline Execution Complete")