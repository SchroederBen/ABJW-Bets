import subprocess
import tkinter as tk
from tkinter import scrolledtext, ttk
from pathlib import Path
import threading
import queue


def extract_human_predictions(full_output: str):
    marker = "=== Human Readable Predictions ==="
    if marker not in full_output:
        return []

    section = full_output.split(marker, 1)[1].strip()
    lines = [line.strip() for line in section.splitlines() if line.strip()]
    return lines


def parse_prediction_line(line: str):
    parts = [p.strip() for p in line.split(" | ")]

    data = {
        "matchup": "",
        "bet": "",
        "confidence": "",
        "edge": "",
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
        elif part.startswith("Reason:"):
            data["reason"] = part.replace("Reason:", "", 1).strip()
        elif part.startswith("Risk Flags:"):
            data["risk_flags"] = part.replace("Risk Flags:", "", 1).strip()

    return data


def show_placeholder(message):
    placeholder = tk.Label(
        cards_container,
        text=message,
        font=("Arial", 11),
        fg="#666666",
        bg="#f3f3f3",
    )
    placeholder.pack(pady=20)
    return placeholder


def clear_cards_container():
    for widget in cards_container.winfo_children():
        widget.destroy()


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
    global is_running
    is_running = False

    status_var.set("Ready to run")
    progress_var.set(0)

    clear_cards_container()

    raw_output_box.delete("1.0", tk.END)
    raw_output_box.pack_forget()
    toggle_raw_button.pack_forget()
    toggle_raw_button.config(text="Show Raw Output")

    progress_bar.pack_forget()

    show_placeholder('Click "Run Predictions" to generate today\'s recommendations.')


def toggle_raw_output():
    if raw_output_box.winfo_ismapped():
        raw_output_box.pack_forget()
        toggle_raw_button.config(text="Show Raw Output")
    else:
        raw_output_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        toggle_raw_button.config(text="Hide Raw Output")


def make_card(parent, prediction):
    card = tk.Frame(parent, bd=1, relief=tk.SOLID, padx=12, pady=10, bg="white")
    card.pack(fill=tk.X, padx=12, pady=8)

    matchup_label = tk.Label(
        card,
        text=prediction["matchup"],
        font=("Arial", 14, "bold"),
        bg="white",
        anchor="w",
    )
    matchup_label.pack(fill=tk.X)

    bet_text = f'Bet: {prediction["bet"]}'
    bet_color = "#1f7a1f" if prediction["bet"].upper() != "PASS" else "#9a7d0a"

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


def update_progress_from_line(line: str):
    text = line.strip()

    if "=== Head-to-Head Stats ===" in text:
        set_stage_status("Building matchup stats")
        progress_var.set(35)
    elif "=== AI Predictions ===" in text:
        set_stage_status("Generating AI predictions")
        progress_var.set(65)
    elif "Saved run results to CSV" in text:
        set_stage_status("Saving results")
        progress_var.set(85)
    elif "=== V1 vs V2 Comparison ===" in text:
        set_stage_status("Comparing model versions")
        progress_var.set(92)
    elif "=== Human Readable Predictions ===" in text:
        set_stage_status("Formatting final output")
        progress_var.set(100)


def run_predictions():
    global is_running, dot_index

    run_button.config(state=tk.DISABLED)
    clear_button.config(state=tk.DISABLED)

    clear_cards_container()

    raw_output_box.delete("1.0", tk.END)
    raw_output_box.pack_forget()
    toggle_raw_button.pack_forget()
    toggle_raw_button.config(text="Show Raw Output")

    is_running = True
    dot_index = 0
    set_stage_status("Starting pipeline")
    progress_var.set(10)

    progress_bar.pack(pady=(0, 10), after=status_label)
    animate_status()

    worker = threading.Thread(target=run_predictions_worker, daemon=True)
    worker.start()
    root.after(100, poll_output_queue)


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
    global is_running
    is_running = False

    run_button.config(state=tk.NORMAL)
    clear_button.config(state=tk.NORMAL)

    toggle_raw_button.pack(pady=(0, 8))

    if return_code != 0:
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
        return

    lines = extract_human_predictions(full_output)

    if not lines:
        status_var.set("Run completed, but no predictions were found")
        clear_cards_container()
        show_placeholder("No readable predictions found.")
        return

    parsed_predictions = [parse_prediction_line(line) for line in lines]

    clear_cards_container()

    results_header = tk.Label(
        cards_container,
        text="Today's Recommendations",
        font=("Arial", 14, "bold"),
        bg="#f3f3f3",
        anchor="w",
    )
    results_header.pack(fill=tk.X, padx=12, pady=(4, 4))

    for prediction in parsed_predictions:
        make_card(cards_container, prediction)

    status_var.set(f"Run completed: {len(parsed_predictions)} prediction(s)")
    progress_var.set(100)


root = tk.Tk()
root.title("ABJW Bets")
root.geometry("1050x760")
root.configure(bg="#f3f3f3")

output_queue = queue.Queue()

is_running = False
current_status_base = "Ready to run"
dot_index = 0
dot_states = [".", "..", "..."]

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

clear_button = tk.Button(
    button_row,
    text="Clear",
    font=("Arial", 11),
    padx=12,
    pady=6,
    command=clear_predictions
)
clear_button.pack(side=tk.LEFT, padx=6)

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

cards_frame_outer = tk.Frame(root, bg="#f3f3f3")
cards_frame_outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

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

show_placeholder('Click "Run Predictions" to generate today\'s recommendations.')

root.mainloop()