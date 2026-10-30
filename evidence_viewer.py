"""Display saved execution and test logs in a small read-only desktop window."""

import argparse
import tkinter as tk
from tkinter import ttk
from src.corpus import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tab", choices=["execution", "tests"], default="execution")
    args = parser.parse_args()
    window = tk.Tk()
    window.title("LumenBridge - Recorded Python Execution")
    window.geometry("1380x1000+40+30")
    window.configure(background="#112638")
    tk.Label(window, text="LUMENBRIDGE / EXECUTION EVIDENCE", font=("Segoe UI", 22, "bold"),
             foreground="#60d6c9", background="#112638", pady=18).pack()
    tk.Label(window, text="Saved output from the included Python scripts. Synthetic classroom demonstration.",
             font=("Segoe UI", 12), foreground="white", background="#112638", pady=8).pack()
    tabs = ttk.Notebook(window)
    tabs.pack(expand=True, fill="both", padx=18, pady=18)
    for label, filename in [("Prototype execution", "execution.txt"), ("Automated checks", "tests.txt")]:
        frame = ttk.Frame(tabs)
        tabs.add(frame, text=label)
        text = tk.Text(frame, font=("Consolas", 13), foreground="white", background="#112638",
                       wrap="word", padx=16, pady=16)
        text.pack(expand=True, fill="both")
        path = ROOT / "evidence" / filename
        text.insert("1.0", path.read_text(encoding="utf-8") if path.exists() else f"Missing log: {path.name}")
        text.configure(state="disabled")
    tabs.select(0 if args.tab == "execution" else 1)
    window.mainloop()


if __name__ == "__main__":
    main()
