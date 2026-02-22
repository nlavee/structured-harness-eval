import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import os
import sys
import re
from rich.logging import RichHandler

# Ensure the parent directory is in the path to allow absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load API keys from .env if present
load_dotenv()

from research_harness.llm_utils import get_llm_kwargs

# Configure logging to write to STDOUT so the orchestrator can capture it
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=None, show_time=False, show_path=False, markup=True)]
)
logger = logging.getLogger("Synthesizer")

import litellm
import sys

# Add current directory to path so we can import schema
sys.path.append(str(Path(__file__).parent))
from schema import AggregatedData



def generate_prompt_context(payload: dict) -> str:
    """Formats the JSON payload into readable text for the LLM."""
    context = []
    
    context.append(f"TOTAL PAIRED SAMPLES ANALYZED: {payload['metadata']['paired_sample_n']}")
    context.append(f"SYSTEMS COMPARED: {', '.join(payload['metadata']['systems'])}\n")
    
    context.append("GLOBAL STATISTICS:")
    for sys, stats in payload['global_statistics'].items():
        context.append(f"\nSystem: {sys}")
        for metric, m_stats in stats.items():
            if m_stats:
                mean = m_stats.get("mean", 0)
                ci_low = m_stats.get("ci_low", 0)
                ci_high = m_stats.get("ci_high", 0)
                context.append(f"  - {metric}: {mean:.2f} (95% CI: [{ci_low:.2f}, {ci_high:.2f}])")
                
    context.append("\nDOMAIN STATISTICS (Per-System Breakdown):")
    domain_stats = payload.get("domain_statistics", {})
    if domain_stats:
        all_domains = set()
        for sys_stats in domain_stats.values():
            all_domains.update(sys_stats.keys())
            
        for domain in sorted(list(all_domains)):
            context.append(f"\n  Domain: {domain}")
            for sys in payload['metadata']['systems']:
                dstats = domain_stats.get(sys, {}).get(domain, {})
                js = dstats.get("judge_score", {}).get("mean", "N/A")
                em = dstats.get("exact_match", {}).get("mean", "N/A")
                sr = dstats.get("soft_recall", {}).get("mean", "N/A")
                hr = dstats.get("hallucination_rate", {}).get("mean", "N/A")
                if js != "N/A":
                    # Format numbers nicely if they aren't N/A
                    em_str = f"{em:.2f}" if isinstance(em, float) else em
                    sr_str = f"{sr:.2f}" if isinstance(sr, float) else sr
                    hr_str = f"{hr:.2f}" if isinstance(hr, float) else hr
                    context.append(f"    - {sys}: Judge Score: {js:.2f}, Exact Match: {em_str}, Soft Recall: {sr_str}, Hallucination: {hr_str}")
    else:
        context.append("  No domain statistics available.")
        
    context.append("\nPAIRWISE WIN RATES (Row > Column on Judge Score):")
    win_rates = payload.get("win_rate_matrix", {})
    if win_rates:
        js_wins = win_rates.get("judge_score", {})
        for row, cols in js_wins.items():
            for col, rate in cols.items():
                if rate > 0:
                    context.append(f"  - {row} beats {col} in {rate*100:.1f}% of samples")
    else:
        context.append("  No win rate data available.")
        
    context.append("\nSTATISTICAL SIGNIFICANCE (Holm-Bonferroni Corrected p-values):")
    p_vals = payload.get("pairwise_significance", {})
    if p_vals:
        js_pvals = p_vals.get("judge_score", {})
        for pair, p in js_pvals.items():
            sig = "**SIGNIFICANT**" if p < 0.05 else "not significant"
            context.append(f"  - {pair}: p={p:.4f} ({sig})")
    else:
        context.append("  No significance data available.")
        
    context.append("\nPAIRED DIVERGENCES (Judge Score = 1.0 vs 0.0):")
    divergences = payload.get("divergence_pairs_ap_rh4", [])
    
    if not divergences:
        context.append("No crisp binary correctness divergences found in this dataset.")
    else:
        # Cap at 20 to avoid blowing up context window
        for i, div in enumerate(divergences[:20]):
            context.append(f"\n--- Divergence Sample {div['sample_id']} ({div['domain']}) ---")
            for sys in payload['metadata']['systems']:
                js = div.get(f"{sys}_judge_score")
                hr = div.get(f"{sys}_hallucination_rate")
                v = div.get(f"{sys}_verbosity")
                context.append(f"  {sys}: Judge= {js}, Hallucination= {hr}, Verbosity= {v}")
                
    return "\n".join(context)

def main():
    parser = argparse.ArgumentParser(description="LLM Qualitative Synthesizer (AP-RH4).")
    parser.add_argument("--data", required=True, help="Path to aggregated_data.json")
    parser.add_argument("--provider", default="anthropic", help="The LiteLLM provider (e.g., anthropic, gemini, openai).")
    parser.add_argument("--model", default="claude-3-5-sonnet-20240620", help="The exact model string (e.g., claude-3-5-sonnet-20240620, gemini-2.5-pro).")
    parser.add_argument("--out-dir", default="research_insights", help="Directory for output.")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.data, "r") as f:
        raw_json = f.read()
        
    try:
        validated_payload = AggregatedData.model_validate_json(raw_json)
        payload = validated_payload.model_dump()
    except Exception as e:
        logger.error(f"FATAL: The incoming aggregated data failed Pydantic schema validation: {e}")
        sys.exit(1)
        
    context_str = generate_prompt_context(payload)
    
    prompt_file = Path(__file__).parent / "prompts" / "synthesizer_prompt.txt"
    with open(prompt_file, "r") as pf:
        prompt_template = pf.read()
        
    prompt = prompt_template.format(data=context_str)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"synthesizer_thought_{timestamp}.txt"
    out_file = out_dir / f"insights_{timestamp}.md"
    
    # AP-RH4 Logging requirements: Log exactly what we sent to the LLM
    logger.info(f"Writing Synthesizer Thought context to {log_file}")
    with open(log_file, "w") as f:
        f.write("=== PROMPT SENT TO LLM ===\n")
        f.write(prompt)
        
    model_uri = f"{args.provider}/{args.model}"
    logger.info(f"Invoking {model_uri} via LiteLLM...")
    
    try:
        base_kwargs = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2 # low temperature for analytical task
        }
        
        # Inject scalable parameters (API keys, reasoning_effort) via the util
        kwargs = get_llm_kwargs(args.provider, args.model, base_kwargs)
        
        response = litellm.completion(**kwargs)
        
        insight_text = response.choices[0].message.content
        
        # Parse out the <thought> block from the response
        thought_match = re.search(r'<thought>(.*?)</thought>', insight_text, re.DOTALL)
        if thought_match:
            thought_content = thought_match.group(1).strip()
            # Remove the thought block from the final insight text
            insight_text = re.sub(r'<thought>.*?</thought>\s*', '', insight_text, flags=re.DOTALL).strip()
            
            with open(log_file, "a") as f:
                f.write("\n\n=== LLM THOUGHT PROCESS ===\n")
                f.write(thought_content)
                
            logger.info("Successfully extracted and logged <thought> block.")
        else:
            logger.warning("No <thought> block found in LLM response.")
            
        with open(out_file, "w") as f:
            f.write(f"# QUALITATIVE SYNTHESIS (Generated by {model_uri})\n\n")
            f.write(insight_text)
            
        logger.info(f"Successfully generated qualitative synthesis at {out_file}")
        
    except Exception as e:
        logger.error(f"Failed to generate LLM insights: {e}")
        raise

if __name__ == "__main__":
    main()
