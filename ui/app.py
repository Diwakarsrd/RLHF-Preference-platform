"""
RLHF Platform - Gradio UI
==========================

Tabs:
  1.  Prompt Manager     - upload or manually add prompts
  2.  Generate Pairs     - produce A/B response pairs
  3.   Annotate           - side-by-side preference annotation
  4.  Metrics            - agreement, Cohen's Kappa, distributions
  5.  Export             - download RLHF JSONL or CSV

Run with:
    python ui/app.py
(Ensure the FastAPI backend is running on localhost:8000)
"""

import os
import json
import time
import requests
import pandas as pd
import gradio as gr

API = os.getenv("BACKEND_URL", "http://localhost:8000")


# 
# API helpers
# 

def api(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API}{path}", timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Is it running?"}
    except Exception as e:
        return {"error": str(e)}


# 
# Tab 1: Prompt Manager
# 

def upload_prompts_file(file, category):
    if file is None:
        return "No file selected."
    with open(file.name, "rb") as f:
        resp = requests.post(
            f"{API}/prompts/upload",
            files={"file": (os.path.basename(file.name), f)},
        )
    if resp.ok:
        data = resp.json()
        return f"Uploaded {data['uploaded']} prompts from {data['source']}"
    return f"Error: {resp.text}"


def add_prompt_manual(text, category):
    if not text.strip():
        return "Prompt text is empty.", refresh_prompt_table()
    resp = api("POST", "/prompts", json={"text": text.strip(), "category": category})
    if "error" in resp:
        return f"{resp['error']}", refresh_prompt_table()
    return f"Prompt #{resp['id']} added.", refresh_prompt_table()


def refresh_prompt_table():
    data = api("GET", "/prompts", params={"limit": 200})
    if isinstance(data, list) and data:
        df = pd.DataFrame(data)[["id", "category", "text"]]
        df["text"] = df["text"].str[:80] + "..."
        return df
    return pd.DataFrame(columns=["id", "category", "text"])


def get_prompt_dropdown():
    data = api("GET", "/prompts", params={"limit": 200})
    if isinstance(data, list):
        return [(f"#{p['id']} - {p['text'][:60]}", p["id"]) for p in data]
    return []


# 
# Tab 2: Generate Pairs
# 

def generate_pair(prompt_choice, temp_a, temp_b, max_tokens):
    if prompt_choice is None:
        return "Select a prompt first.", "", "", ""
    resp = api("POST", "/generate", json={
        "prompt_id":      int(prompt_choice),
        "temp_a":         temp_a,
        "temp_b":         temp_b,
        "max_new_tokens": int(max_tokens),
    })
    if "error" in resp:
        return f"{resp['error']}", "", "", ""
    return (
        f"Pair #{resp['id']} generated.",
        resp.get("prompt", ""),
        resp.get("response_a", ""),
        resp.get("response_b", ""),
    )


def refresh_pair_table():
    data = api("GET", "/pairs", params={"limit": 100})
    if isinstance(data, list) and data:
        df = pd.DataFrame(data)[["id", "prompt_id", "model_a", "model_b"]]
        return df
    return pd.DataFrame(columns=["id", "prompt_id", "model_a", "model_b"])


def get_pair_dropdown():
    data = api("GET", "/pairs", params={"limit": 200})
    if isinstance(data, list):
        return [(f"Pair #{p['id']}", p["id"]) for p in data]
    return []


# 
# Tab 3: Annotate
# 

_annotation_start = [time.time()]


def load_pair_for_annotation(pair_choice):
    if pair_choice is None:
        return "", "", "", ""
    resp = api("GET", f"/pairs/{int(pair_choice)}")
    if "error" in resp:
        return f"Error: {resp['error']}", "", "", ""
    _annotation_start[0] = time.time()
    return (
        resp.get("prompt", ""),
        resp.get("response_a", ""),
        resp.get("response_b", ""),
        "",   # clear status
    )


def submit_annotation(pair_choice, annotator_id, preference, reasoning, confidence):
    if not pair_choice:
        return "Select a pair."
    if not annotator_id.strip():
        return "Enter your annotator ID."
    if not preference:
        return "Select a preference."

    elapsed = round(time.time() - _annotation_start[0], 2)
    resp = api("POST", "/annotations", json={
        "pair_id":        int(pair_choice),
        "annotator_id":   annotator_id.strip(),
        "preference":     preference,
        "reasoning":      reasoning,
        "confidence":     int(confidence),
        "time_spent_sec": elapsed,
    })
    if "error" in resp:
        return f"{resp['error']}"
    return f"Annotation #{resp['id']} saved! (took {elapsed}s)"


# 
# Tab 4: Metrics
# 

def load_metrics():
    m = api("GET", "/metrics")
    if "error" in m:
        return f" {m['error']}", None, None

    dist = m.get("preference_distribution", {})
    kappa = m.get("cohens_kappa")
    iaa   = m.get("raw_iaa")

    summary = f"""
##  Annotation Summary

| Metric | Value |
|--------|-------|
| Total Annotations | {m['total_annotations']} |
| Unique Pairs | {m['unique_pairs']} |
| Unique Annotators | {m['unique_annotators']} |
| Raw IAA | {f"{iaa:.1%}" if iaa is not None else "Need >=2 annotators"} |
| Cohen's Kappa | {f"{kappa:.3f}" if kappa is not None else "N/A"} |
| Kappa Interpretation | {m.get('kappa_interpretation', 'N/A')} |
| Conf-Weighted Agreement | {f"{m['confidence_weighted_agreement']:.1%}" if m.get('confidence_weighted_agreement') else "N/A"} |

### Preference Distribution
-  Model A preferred: **{dist.get('A', 0)}%**
-  Model B preferred: **{dist.get('B', 0)}%**
-  Tie: **{dist.get('tie', 0)}%**

### Annotator Consistency
{_format_consistency(m.get('consistency', {}))}
    """

    # Preference bar chart data
    dist_df = pd.DataFrame([
        {"Label": "A Better", "Pct": dist.get("A", 0)},
        {"Label": "B Better", "Pct": dist.get("B", 0)},
        {"Label": "Tie",      "Pct": dist.get("tie", 0)},
    ])

    return summary.strip(), dist_df, None


def _format_consistency(consistency_dict):
    if not consistency_dict:
        return "_No repeated annotations yet._"
    lines = [f"- **{ann}**: {v:.1%}" for ann, v in consistency_dict.items()]
    return "\n".join(lines)


# 
# Tab 5: Export
# 

def export_rlhf_preview():
    """Return first 5 lines of the RLHF export for preview."""
    try:
        r = requests.get(f"{API}/export/rlhf", timeout=15)
        r.raise_for_status()
        lines = r.text.strip().split("\n")[:5]
        preview = []
        for line in lines:
            obj = json.loads(line)
            preview.append({
                "prompt":   obj["prompt"][:60] + "...",
                "chosen":   obj["chosen"][:60]  + "...",
                "rejected": obj["rejected"][:60] + "...",
            })
        return pd.DataFrame(preview), f" {len(lines)} records (showing 5)"
    except Exception as e:
        return pd.DataFrame(), f" {e}"


def download_rlhf():
    try:
        r = requests.get(f"{API}/export/rlhf", timeout=15)
        path = "/tmp/rlhf_dataset.jsonl"
        with open(path, "w") as f:
            f.write(r.text)
        return path, "Ready to download"
    except Exception as e:
        return None, f" {e}"


def download_csv():
    try:
        r = requests.get(f"{API}/export/csv", timeout=15)
        path = "/tmp/annotations.csv"
        with open(path, "w") as f:
            f.write(r.text)
        return path, "Ready to download"
    except Exception as e:
        return None, f" {e}"


# 
# Build Gradio App
# 

with gr.Blocks(
    title="RLHF Preference Platform",
    theme=gr.themes.Soft(primary_hue="indigo"),
    css="""
    .metric-box { border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; }
    .tab-header { font-weight: 700; font-size: 1.1em; }
    """,
) as demo:

    gr.Markdown(
        """
        #  RLHF Preference & Reward Modeling Platform
        > Generate, annotate, and export preference data for LLM fine-tuning
        """,
        elem_classes="tab-header",
    )

    #  Tab 1: Prompt Manager 
    with gr.Tab("Prompt Manager"):
        gr.Markdown("### Load Prompts")
        with gr.Row():
            with gr.Column():
                file_input   = gr.File(label="Upload JSON or CSV", file_types=[".json", ".csv"])
                upload_btn   = gr.Button("Upload File", variant="primary")
                upload_status = gr.Textbox(label="Status", interactive=False)

            with gr.Column():
                manual_text = gr.Textbox(label="Or add a single prompt", lines=3,
                                          placeholder="Enter your prompt here...")
                manual_cat  = gr.Dropdown(
                    choices=["general", "math", "coding", "reasoning"],
                    value="general", label="Category",
                )
                add_btn    = gr.Button("Add Prompt")
                add_status = gr.Textbox(label="Status", interactive=False)

        prompt_table = gr.Dataframe(label="Loaded Prompts", interactive=False)
        refresh_btn  = gr.Button("Refresh Table")

        upload_btn.click(upload_prompts_file, [file_input, manual_cat], upload_status)
        add_btn.click(add_prompt_manual, [manual_text, manual_cat], [add_status, prompt_table])
        refresh_btn.click(refresh_prompt_table, [], prompt_table)

    #  Tab 2: Generate Pairs 
    with gr.Tab("Generate Pairs"):
        gr.Markdown("### Generate A/B Response Pairs")
        with gr.Row():
            with gr.Column(scale=1):
                prompt_dd    = gr.Dropdown(label="Select Prompt", choices=[], interactive=True)
                refresh_dd   = gr.Button("Refresh Prompts")
                temp_a_sl    = gr.Slider(0.1, 2.0, value=0.7,  step=0.1, label="Temperature A (focused)")
                temp_b_sl    = gr.Slider(0.1, 2.0, value=1.2,  step=0.1, label="Temperature B (creative)")
                max_tok_sl   = gr.Slider(64,  512,  value=256, step=32,  label="Max New Tokens")
                gen_btn      = gr.Button("Generate", variant="primary")
                gen_status   = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                prompt_disp  = gr.Textbox(label="Prompt",     lines=3, interactive=False)
                with gr.Row():
                    resp_a = gr.Textbox(label="Response A (low temp)",  lines=8, interactive=False)
                    resp_b = gr.Textbox(label="Response B (high temp)", lines=8, interactive=False)

        pair_table = gr.Dataframe(label="Generated Pairs", interactive=False)
        ref_pairs  = gr.Button("Refresh Pairs")

        refresh_dd.click(lambda: gr.update(choices=get_prompt_dropdown()), [], prompt_dd)
        gen_btn.click(generate_pair,
                      [prompt_dd, temp_a_sl, temp_b_sl, max_tok_sl],
                      [gen_status, prompt_disp, resp_a, resp_b])
        ref_pairs.click(refresh_pair_table, [], pair_table)

    #  Tab 3: Annotate 
    with gr.Tab("Annotate"):
        gr.Markdown("### Preference Annotation Interface")
        with gr.Row():
            pair_dd_ann  = gr.Dropdown(label="Select Pair to Annotate", choices=[], interactive=True)
            ann_refresh  = gr.Button("Refresh Pairs")
            annotator_id = gr.Textbox(label="Your Annotator ID", placeholder="e.g. alice@lab.ai")
            load_pair_btn = gr.Button("Load Pair", variant="secondary")

        prompt_ann = gr.Textbox(label="Prompt", lines=2, interactive=False)
        with gr.Row():
            resp_a_ann = gr.Textbox(label="Response A", lines=10, interactive=False)
            resp_b_ann = gr.Textbox(label="Response B", lines=10, interactive=False)

        gr.Markdown("### Your Judgment")
        with gr.Row():
            preference = gr.Radio(
                choices=[("A is Better", "A"), ("B is Better", "B"), ("Tie", "tie")],
                label="Preference",
            )
            confidence = gr.Slider(1, 5, value=3, step=1, label="Confidence (1=low, 5=high)")

        reasoning  = gr.Textbox(label="Reasoning (optional)", lines=2,
                                 placeholder="Why did you choose this? Any interesting differences?")
        submit_btn = gr.Button("Submit Annotation", variant="primary")
        ann_status = gr.Textbox(label="Status", interactive=False)

        ann_refresh.click(lambda: gr.update(choices=get_pair_dropdown()), [], pair_dd_ann)
        load_pair_btn.click(load_pair_for_annotation, [pair_dd_ann],
                            [prompt_ann, resp_a_ann, resp_b_ann, ann_status])
        submit_btn.click(submit_annotation,
                         [pair_dd_ann, annotator_id, preference, reasoning, confidence],
                         ann_status)

    #  Tab 4: Metrics 
    with gr.Tab("Metrics"):
        gr.Markdown("### Agreement & Quality Metrics")
        calc_btn = gr.Button("Calculate Metrics", variant="primary")
        metrics_md = gr.Markdown()
        dist_chart = gr.BarPlot(
            x="Label", y="Pct", title="Preference Distribution",
            y_lim=[0, 100], color="Label",
            visible=True,
        )

        calc_btn.click(load_metrics, [], [metrics_md, dist_chart, gr.State()])

    #  Tab 5: Export 
    with gr.Tab("Export"):
        gr.Markdown("### Export RLHF Dataset")
        preview_btn  = gr.Button("Preview RLHF Export")
        preview_df   = gr.Dataframe(label="Preview (first 5 rows)", interactive=False)
        preview_stat = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("#### Downloads")
        with gr.Row():
            rlhf_btn    = gr.Button("Download RLHF JSONL", variant="primary")
            rlhf_file   = gr.File(label="RLHF Dataset (.jsonl)")
            rlhf_status = gr.Textbox(interactive=False)

            csv_btn     = gr.Button("Download Annotations CSV")
            csv_file    = gr.File(label="Annotations (.csv)")
            csv_status  = gr.Textbox(interactive=False)

        preview_btn.click(export_rlhf_preview, [], [preview_df, preview_stat])
        rlhf_btn.click(download_rlhf, [], [rlhf_file, rlhf_status])
        csv_btn.click(download_csv, [], [csv_file, csv_status])

    # Initial load
    demo.load(refresh_prompt_table, [], prompt_table)
    demo.load(refresh_pair_table,   [], pair_table)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
