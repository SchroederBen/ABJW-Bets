import subprocess
import tkinter as tk
from tkinter import scrolledtext, ttk
from pathlib import Path
import threading
import queue
import re


def extract_prediction_blocks(full_output: str):
    if "=== Human Readable Predictions ===" in full_output:
        section = full_output.split("=== Human Readable Predictions ===", 1)[1].strip()
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        return ("human_readable", lines)

    if "NBA SPREAD PICKS" in full_output:
        section = full_output.split("NBA SPREAD PICKS", 1)[1]

        raw_blocks = re.split(r"\n\s*\d+\.\s+", section)
        parsed_blocks = []

        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue
            if "Pick:" not in block:
                continue
            parsed_blocks.append(block)

        return ("spread_blocks", parsed_blocks)

    return ("none", [])


def parse_human_prediction_line(line: str):
    parts = [p.strip() for p in line.split(" | ")]

    data = {
        "matchup": "",
        "bet": "",
        "confidence": "",
        "edge": "",
        "projected_margin": "",
        "fair_home_spread": "",
        "reason": "",
        "risk_flags": "",
    }

    if parts:
        data["matchup"] = parts[0]

    for part in parts[1:]:
        if part.startswith("Bet:"):
            data["bet"] = part.replace("Bet:", "", 1).strip()
        elif part.startswith("Confidence:"):
            data["confidence"] = part.replace("Confidence:", "", 1).strip()
        elif part.startswith("Estimated Edge:"):
            data["edge"] = part.replace("Estimated Edge:", "", 1).strip()
        elif part.startswith("Projected Margin:"):
            data["projected_margin"] = part.replace("Projected Margin:", "", 1).strip()
        elif part.startswith("Fair Home Spread:"):
            data["fair_home_spread"] = part.replace("Fair Home Spread:", "", 1).strip()
        elif part.startswith("Reason:"):
            data["reason"] = part.replace("Reason:", "", 1).strip()
        elif part.startswith("Risk Flags:"):
            data["risk_flags"] = part.replace("Risk Flags:", "", 1).strip()

    return data


def parse_spread_pick_block(block: str):
    data = {
        "matchup": "",
        "bet": "",
        "confidence": "",
        "edge": "",
        "projected_margin": "",
        "fair_home_spread": "",
        "reason": "",
        "risk_flags": "",
    }

    lines = [line.strip() for line in block.splitlines() if line.strip()]

    if lines:
        data["matchup"] = lines[0]

    def extract_value(label):
        pattern = rf"{re.escape(label)}\s*(.+)"
        match = re.search(pattern, block)
        return match.group(1).strip() if match else ""

    data["bet"] = extract_value("Pick:")
    data["confidence"] = extract_value("Confidence:")
    data["edge"] = extract_value("Estimated Edge:")
    data["projected_margin"] = extract_value("Projected Margin:")
    data["fair_home_spread"] = extract_value("Fair Home Spread:")
    data["reason"] = extract_value("Reason:")
    data["risk_flags"] = extract_value("Risk Flags:")

    return data


def format_predictions_for_clipboard(predictions):
    lines = []

    for index, prediction in enumerate(predictions, start=1):
        risk_text = prediction["risk_flags"] if prediction["risk_flags"] else "None"
        projected_margin = prediction["projected_margin"] if prediction["projected_margin"] else "N/A"
        fair_home_spread = prediction["fair_home_spread"] if prediction["fair_home_spread"] else "N/A"

        lines.append(f"{index}. {prediction['matchup']}")
        lines.append(f"Bet: {prediction['bet']}")
        lines.append(f"Confidence: {prediction['confidence']}")
        lines.append(f"Estimated Edge: {prediction['edge']}")
        lines.append(f"Projected Margin: {projected_margin}")
        lines.append(f"Fair Home Spread: {fair_home_spread}")
        lines.append(f"Reason: {prediction['reason']}")
        lines.append(f"Risk Flags: {risk_text}")
        lines.append("")

    return "\n".join(lines).strip()


def copy_results():
    if not last_predictions:
        status_var.set("No results to copy")
        return

    text_to_copy = format_predictions_for_clipboard(last_predictions)
    root.clipboard_clear()
    root.clipboard_append(text_to_copy)
    root.update()
    status_var.set("Results copied to clipboard")


def on_results_mousewheel(event):
    if not results_canvas.winfo_exists():
        return

    bbox = results_canvas.bbox("all")
    if not bbox:
        return

    visible_height = results_canvas.winfo_height()
    content_height = bbox[3] - bbox[1]

    if content_height <= visible_height:
        return

    if event.num == 4:
        results_canvas.yview_scroll(-1, "units")
    elif event.num == 5:
        results_canvas.yview_scroll(1, "units")
    elif event.delta:
        results_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


def bind_mousewheel_recursive(widget):
    widget.bind("<MouseWheel>", on_results_mousewheel)
    widget.bind("<Button-4>", on_results_mousewheel)
    widget.bind("<Button-5>", on_results_mousewheel)

    for child in widget.winfo_children():
        bind_mousewheel_recursive(child)


def refresh_scroll_region():
    results_canvas.update_idletasks()
    results_canvas.configure(scrollregion=results_canvas.bbox("all"))


def show_placeholder(message):
    placeholder = tk.Label(
        cards_container,
        text=message,
        font=("Arial", 11),
        fg="#666666",
        bg="#f3f3f3",
    )
    placeholder.pack(pady=20)
    bind_mousewheel_recursive(placeholder)
    refresh_scroll_region()
    return placeholder


def clear_cards_container():
    for widget in cards_container.winfo_children():
        widget.destroy()
    refresh_scroll_region()


def set_stage_status(text):
    global current_status_base
    current_status_base = text


def animate_status():
    global dot_index

    if is_running:
        dots = dot_states[dot_index]
        status_var.set(f"{current_status_base}{dots}")
        dot_index = (dot_index + 1) % len(dot_states)
        root.after(400, animate_status)


def clear_predictions():
    global is_running, last_predictions
    is_running = False
    last_predictions = []

    status_var.set("Ready to run")
    progress_var.set(0)

    clear_cards_container()

    raw_output_box.delete("1.0", tk.END)
    raw_output_box.pack_forget()
    toggle_raw_button.pack_forget()
    toggle_raw_button.config(text="Show Raw Output")

    progress_bar.pack_forget()

    show_placeholder('Click "Run Predictions" or "Run Demo" to generate recommendations.')


def toggle_raw_output():
    if raw_output_box.winfo_ismapped():
        raw_output_box.pack_forget()
        toggle_raw_button.config(text="Show Raw Output")
    else:
        raw_output_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        toggle_raw_button.config(text="Hide Raw Output")


def make_card(parent, prediction):
    is_pass = prediction["bet"].upper() == "PASS"

    if is_pass:
        border_color = "#b08d2f"
        bet_color = "#9a7d0a"
    else:
        border_color = "#2e8b57"
        bet_color = "#1f7a1f"

    outer = tk.Frame(parent, bg=border_color, padx=2, pady=2)
    outer.pack(fill=tk.X, padx=12, pady=8)

    card = tk.Frame(outer, bg="white", padx=12, pady=10)
    card.pack(fill=tk.X)

    matchup_label = tk.Label(
        card,
        text=prediction["matchup"],
        font=("Arial", 14, "bold"),
        bg="white",
        anchor="w",
    )
    matchup_label.pack(fill=tk.X)

    bet_text = f'Bet: {prediction["bet"]}'

    bet_label = tk.Label(
        card,
        text=bet_text,
        font=("Arial", 12, "bold"),
        fg=bet_color,
        bg="white",
        anchor="w",
    )
    bet_label.pack(fill=tk.X, pady=(6, 0))

    details_text = f'Confidence: {prediction["confidence"]}    Estimated Edge: {prediction["edge"]}'
    details_label = tk.Label(
        card,
        text=details_text,
        font=("Arial", 10),
        bg="white",
        anchor="w",
        justify="left",
    )
    details_label.pack(fill=tk.X, pady=(4, 0))

    margin_text = (
        f'Projected Margin: {prediction["projected_margin"]}    '
        f'Fair Home Spread: {prediction["fair_home_spread"]}'
    )
    margin_label = tk.Label(
        card,
        text=margin_text,
        font=("Arial", 10),
        bg="white",
        anchor="w",
        justify="left",
    )
    margin_label.pack(fill=tk.X, pady=(4, 0))

    reason_label = tk.Label(
        card,
        text=f'Reason: {prediction["reason"]}',
        font=("Arial", 10),
        bg="white",
        anchor="w",
        justify="left",
        wraplength=900,
    )
    reason_label.pack(fill=tk.X, pady=(6, 0))

    risk_text = prediction["risk_flags"] if prediction["risk_flags"] else "None"
    risk_label = tk.Label(
        card,
        text=f'Risk Flags: {risk_text}',
        font=("Arial", 10),
        bg="white",
        anchor="w",
        justify="left",
        wraplength=900,
    )
    risk_label.pack(fill=tk.X, pady=(4, 0))

    bind_mousewheel_recursive(outer)
    refresh_scroll_region()


def update_progress_from_line(line: str):
    text = line.strip()

    if "=== DEMO MODE ENABLED ===" in text:
        set_stage_status("Loading demo games")
        progress_var.set(20)
    elif "=== Fetching data ===" in text:
        set_stage_status("Fetching data")
        progress_var.set(15)
    elif "=== Head-to-Head Stats ===" in text:
        set_stage_status("Building matchup stats")
        progress_var.set(35)
    elif "=== L1 Feature Preview ===" in text:
        set_stage_status("Preparing L1 features")
        progress_var.set(55)
    elif "=== AI Predictions ===" in text:
        set_stage_status("Generating AI predictions")
        progress_var.set(65)
    elif "=== Edge Model Comparison ===" in text:
        set_stage_status("Comparing model outputs")
        progress_var.set(80)
    elif "NBA SPREAD PICKS" in text:
        set_stage_status("Formatting final output")
        progress_var.set(100)
    elif "Saved run results to CSV" in text:
        set_stage_status("Saving results")
        progress_var.set(90)
    elif "=== V1 vs V2 Comparison ===" in text:
        set_stage_status("Comparing model versions")
        progress_var.set(92)
    elif "=== Human Readable Predictions ===" in text:
        set_stage_status("Formatting final output")
        progress_var.set(100)


def start_run(mode_label, worker_target):
    global is_running, dot_index, last_predictions
    last_predictions = []

    run_button.config(state=tk.DISABLED)
    demo_button.config(state=tk.DISABLED)
    clear_button.config(state=tk.DISABLED)
    copy_button.config(state=tk.DISABLED)

    clear_cards_container()

    raw_output_box.delete("1.0", tk.END)
    raw_output_box.pack_forget()
    toggle_raw_button.pack_forget()
    toggle_raw_button.config(text="Show Raw Output")

    is_running = True
    dot_index = 0
    set_stage_status(mode_label)
    progress_var.set(10)

    progress_bar.pack(pady=(0, 10), after=status_label)
    animate_status()

    worker = threading.Thread(target=worker_target, daemon=True)
    worker.start()
    root.after(100, poll_output_queue)


def run_predictions():
    start_run("Starting pipeline", run_predictions_worker)


def run_demo_predictions():
    start_run("Starting demo pipeline", run_demo_worker)


def run_predictions_worker():
    repo_root = Path(__file__).parent

    process = subprocess.Popen(
        ["py", "-u", "AI\\main.py"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    collected_output = []

    if process.stdout is not None:
        for line in process.stdout:
            collected_output.append(line)
            output_queue.put(("line", line))

    return_code = process.wait()
    full_output = "".join(collected_output)
    output_queue.put(("done", (return_code, full_output)))


def run_demo_worker():
    repo_root = Path(__file__).parent

    process = subprocess.Popen(
        ["py", "-u", "AI\\main.py", "--demo"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    collected_output = []

    if process.stdout is not None:
        for line in process.stdout:
            collected_output.append(line)
            output_queue.put(("line", line))

    return_code = process.wait()
    full_output = "".join(collected_output)
    output_queue.put(("done", (return_code, full_output)))


def poll_output_queue():
    try:
        while True:
            kind, payload = output_queue.get_nowait()

            if kind == "line":
                raw_output_box.insert(tk.END, payload)
                raw_output_box.see(tk.END)
                update_progress_from_line(payload)

            elif kind == "done":
                return_code, full_output = payload
                finish_run(return_code, full_output)
                return
    except queue.Empty:
        root.after(100, poll_output_queue)


def finish_run(return_code, full_output):
    global is_running, last_predictions
    is_running = False

    run_button.config(state=tk.NORMAL)
    demo_button.config(state=tk.NORMAL)
    clear_button.config(state=tk.NORMAL)

    toggle_raw_button.pack(pady=(0, 8))

    if return_code != 0:
        copy_button.config(state=tk.DISABLED)
        status_var.set("Run failed")
        progress_var.set(0)

        clear_cards_container()

        error_card = tk.Frame(cards_container, bd=1, relief=tk.SOLID, padx=12, pady=10, bg="white")
        error_card.pack(fill=tk.X, padx=12, pady=8)

        tk.Label(
            error_card,
            text="The script failed.",
            font=("Arial", 12, "bold"),
            fg="red",
            bg="white",
            anchor="w",
        ).pack(fill=tk.X)

        tk.Label(
            error_card,
            text="Open raw output below to see the error details.",
            font=("Arial", 10),
            bg="white",
            anchor="w",
        ).pack(fill=tk.X, pady=(4, 0))

        refresh_scroll_region()
        return

    mode, blocks = extract_prediction_blocks(full_output)

    if not blocks:
        copy_button.config(state=tk.DISABLED)
        status_var.set("Run completed, but no predictions were found")
        clear_cards_container()
        show_placeholder("No readable predictions found.")
        return

    if mode == "human_readable":
        parsed_predictions = [parse_human_prediction_line(line) for line in blocks]
    else:
        parsed_predictions = [parse_spread_pick_block(block) for block in blocks]

    last_predictions = parsed_predictions

    clear_cards_container()

    results_header = tk.Label(
        cards_container,
        text="Today's Recommendations",
        font=("Arial", 14, "bold"),
        bg="#f3f3f3",
        anchor="w",
    )
    results_header.pack(fill=tk.X, padx=12, pady=(4, 4))
    bind_mousewheel_recursive(results_header)

    for prediction in parsed_predictions:
        make_card(cards_container, prediction)

    copy_button.config(state=tk.NORMAL)
    status_var.set(f"Run completed: {len(parsed_predictions)} prediction(s)")
    progress_var.set(100)
    refresh_scroll_region()


root = tk.Tk()
root.title("ABJW Bets")
root.geometry("1050x760")
root.configure(bg="#f3f3f3")

output_queue = queue.Queue()

is_running = False
current_status_base = "Ready to run"
dot_index = 0
dot_states = [".", "..", "..."]
last_predictions = []

title_label = tk.Label(
    root,
    text="ABJW Bets",
    font=("Arial", 20, "bold"),
    bg="#f3f3f3",
)
title_label.pack(pady=(12, 4))

subtitle_label = tk.Label(
    root,
    text="NBA betting recommendation tool",
    font=("Arial", 10),
    fg="#555555",
    bg="#f3f3f3",
)
subtitle_label.pack(pady=(0, 10))

button_row = tk.Frame(root, bg="#f3f3f3")
button_row.pack(pady=(0, 10))

run_button = tk.Button(
    button_row,
    text="Run Predictions",
    font=("Arial", 11),
    padx=12,
    pady=6,
    command=run_predictions
)
run_button.pack(side=tk.LEFT, padx=6)

demo_button = tk.Button(
    button_row,
    text="Run Demo",
    font=("Arial", 11),
    padx=12,
    pady=6,
    command=run_demo_predictions
)
demo_button.pack(side=tk.LEFT, padx=6)

clear_button = tk.Button(
    button_row,
    text="Clear",
    font=("Arial", 11),
    padx=12,
    pady=6,
    command=clear_predictions
)
clear_button.pack(side=tk.LEFT, padx=6)

copy_button = tk.Button(
    button_row,
    text="Copy Results",
    font=("Arial", 11),
    padx=12,
    pady=6,
    command=copy_results,
    state=tk.DISABLED
)
copy_button.pack(side=tk.LEFT, padx=6)

status_var = tk.StringVar(value="Ready to run")
status_label = tk.Label(
    root,
    textvariable=status_var,
    font=("Arial", 10, "italic"),
    fg="#444444",
    bg="#f3f3f3",
)
status_label.pack(pady=(0, 4))

progress_var = tk.IntVar(value=0)
progress_bar = ttk.Progressbar(
    root,
    orient="horizontal",
    mode="determinate",
    length=420,
    maximum=100,
    variable=progress_var
)
progress_bar.pack_forget()

results_outer = tk.Frame(root, bg="#f3f3f3")
results_outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

results_canvas = tk.Canvas(results_outer, bg="#f3f3f3", highlightthickness=0)
results_scrollbar = ttk.Scrollbar(results_outer, orient="vertical", command=results_canvas.yview)
results_canvas.configure(yscrollcommand=results_scrollbar.set)

results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

cards_frame_outer = tk.Frame(results_canvas, bg="#f3f3f3")
canvas_window = results_canvas.create_window((0, 0), window=cards_frame_outer, anchor="nw")


def on_cards_frame_configure(_event):
    refresh_scroll_region()


def on_canvas_configure(event):
    results_canvas.itemconfigure(canvas_window, width=event.width)


cards_frame_outer.bind("<Configure>", on_cards_frame_configure)
results_canvas.bind("<Configure>", on_canvas_configure)
bind_mousewheel_recursive(results_canvas)

cards_container = tk.Frame(cards_frame_outer, bg="#f3f3f3")
cards_container.pack(fill=tk.BOTH, expand=True)

toggle_raw_button = tk.Button(
    root,
    text="Show Raw Output",
    font=("Arial", 10),
    command=toggle_raw_output
)

raw_output_box = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    font=("Consolas", 9),
    height=12
)

show_placeholder('Click "Run Predictions" or "Run Demo" to generate recommendations.')

root.mainloop()