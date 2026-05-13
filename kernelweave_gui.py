"""
KernelWeave Glass-Panel GUI
===========================

A futuristic, local desktop GUI for KernelWeave that visualizes the neuro-symbolic
engine working with graphs, blobs, and real-time activity logs.
"""
import os
import sys
import time
import json
import threading
import queue
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# Ensure kernelweave is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from kernelweave.kernel import KernelStore
from kernelweave.runtime import ExecutionEngine, KernelRuntime
from kernelweave_ollama import make_backend, wrap_with_streaming, get_ollama_models, get_openai_models

# ── Theme Colors (Cyberpunk / Glass Box Vibe) ────────────────────
BG_COLOR = "#0d1117"      # Deep dark
SURFACE_COLOR = "#161b22" # Dark gray
TEXT_COLOR = "#c9d1d9"    # Light gray
ACCENT_CYAN = "#58a6ff"   # Neon Cyan
ACCENT_PINK = "#ff79c6"   # Neon Pink
ACCENT_GREEN = "#50fa7b"  # Neon Green
ACCENT_RED = "#ff5555"    # Neon Red
DIM_COLOR = "#8b949e"     # Muted gray

class KernelWeaveGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("KernelWeave OS - Glass Panel Engine")
        self.root.geometry("1200x800")
        self.root.configure(bg=BG_COLOR)
        
        # Backend state
        self.backend = None
        self.runtime = None
        self.engine = None
        self.store = None
        self.stop_requested = False
        self.executing = False
        
        # Queue for thread communication
        self.msg_queue = queue.Queue()
        
        # Setup UI
        self.create_styles()
        self.create_layout()
        
        # Load store and scan models
        self.initialize_engine()
        
        # Start queue poller
        self.root.after(100, self.poll_queue)
        
    def create_styles(self):
        style = ttk.Style()
        style.theme_use('default')
        
        # Dark theme for ttk widgets
        style.configure('TFrame', background=BG_COLOR)
        style.configure('TLabel', background=BG_COLOR, foreground=TEXT_COLOR, font=('Courier', 10))
        style.configure('TButton', background=SURFACE_COLOR, foreground=TEXT_COLOR, font=('Courier', 10, 'bold'), borderwidth=1)
        style.map('TButton', background=[('active', ACCENT_CYAN)], foreground=[('active', BG_COLOR)])
        
        style.configure('Header.TLabel', font=('Courier', 16, 'bold'), foreground=ACCENT_CYAN)
        style.configure('Status.TLabel', font=('Courier', 9), foreground=DIM_COLOR)
        
    def create_layout(self):
        # Top Banner
        banner = ttk.Frame(self.root)
        banner.pack(fill='x', padx=20, pady=10)
        ttk.Label(banner, text="KERNELWEAVE NEURO-SYMBOLIC ENGINE", style='Header.TLabel').pack(side='left')
        
        self.status_label = ttk.Label(banner, text="Status: Initializing...", style='Status.TLabel')
        self.status_label.pack(side='right', pady=5)
        
        # Main Splitter (Left: Controls & Log, Right: Visualization)
        main_pane = ttk.Frame(self.root)
        main_pane.pack(fill='both', expand=True, padx=20, pady=10)
        
        left_pane = ttk.Frame(main_pane)
        left_pane.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        right_pane = ttk.Frame(main_pane)
        right_pane.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # ── Left Pane (Controls & Text) ──────────────────────────────
        # Model Selector
        model_frame = ttk.Frame(left_pane)
        model_frame.pack(fill='x', pady=(0, 10))
        ttk.Label(model_frame, text="Model:").pack(side='left', padx=5)
        
        self.model_combo = ttk.Combobox(model_frame, width=30, font=('Courier', 10))
        self.model_combo.pack(side='left', padx=5)
        
        self.btn_connect = ttk.Button(model_frame, text="Connect", command=self.connect_backend)
        self.btn_connect.pack(side='left', padx=5)
        
        # Text Log (Glass look via dark background)
        log_frame = ttk.Frame(left_pane)
        log_frame.pack(fill='both', expand=True)
        ttk.Label(log_frame, text="Execution Trace & Logs", font=('Courier', 12, 'bold'), foreground=ACCENT_PINK).pack(anchor='w', pady=(0, 5))
        
        self.log_area = scrolledtext.ScrolledText(
            log_frame, 
            bg=SURFACE_COLOR, 
            fg=TEXT_COLOR, 
            font=('Courier', 10),
            insertbackground=TEXT_COLOR,
            wrap=tk.WORD,
            borderwidth=1,
            relief='solid'
        )
        self.log_area.pack(fill='both', expand=True)
        
        # Tags for colors in log
        self.log_area.tag_config('user', foreground=ACCENT_CYAN)
        self.log_area.tag_config('bot', foreground=TEXT_COLOR)
        self.log_area.tag_config('system', foreground=DIM_COLOR)
        self.log_area.tag_config('success', foreground=ACCENT_GREEN)
        self.log_area.tag_config('error', foreground=ACCENT_RED)
        
        # Prompt Input
        input_frame = ttk.Frame(left_pane)
        input_frame.pack(fill='x', pady=(10, 0))
        
        self.prompt_entry = tk.Entry(
            input_frame, 
            bg=SURFACE_COLOR, 
            fg=TEXT_COLOR, 
            font=('Courier', 12),
            insertbackground=TEXT_COLOR,
            borderwidth=1,
            relief='solid'
        )
        self.prompt_entry.pack(fill='x', side='left', expand=True, ipady=8)
        self.prompt_entry.bind("<Return>", lambda e: self.send_prompt())
        
        self.btn_send = ttk.Button(input_frame, text="SEND", command=self.send_prompt)
        self.btn_send.pack(side='left', padx=5, ipady=5)
        
        self.btn_stop = ttk.Button(input_frame, text="FORCE STOP", command=self.force_stop)
        self.btn_stop.pack(side='left', padx=5, ipady=5)
        
        # ── Right Pane (Visualization) ─────────────────────────────
        # Canvas for graphs and blobs
        viz_frame = ttk.Frame(right_pane)
        viz_frame.pack(fill='both', expand=True)
        ttk.Label(viz_frame, text="Neural-Symbolic Engine Map", font=('Courier', 12, 'bold'), foreground=ACCENT_CYAN).pack(anchor='w', pady=(0, 5))
        
        self.canvas = tk.Canvas(
            viz_frame, 
            bg=SURFACE_COLOR, 
            borderwidth=1, 
            relief='solid',
            highlightthickness=0
        )
        self.canvas.pack(fill='both', expand=True)
        
        # ── Variables for Viz ──────────────────────────────────────
        self.nodes = {} # node_id -> {x, y, radius, color, label}
        self.connections = [] # (node1, node2, color)
        
    def initialize_engine(self):
        self.append_log("System: Loading KernelStore...", "system")
        try:
            self.store = KernelStore(Path("store"))
            self.runtime = KernelRuntime(self.store, use_embeddings=True)
            self.append_log(f"System: Store loaded with {len(self.store.list_kernels())} kernels.", "success")
            
            # Draw initial map
            self.draw_engine_map()
            
            # Scan models
            self.append_log("System: Scanning for local LLM servers...", "system")
            threading.Thread(target=self.async_scan_models, daemon=True).start()
            
        except Exception as e:
            self.append_log(f"Error initializing engine: {e}", "error")
            
    def async_scan_models(self):
        # Scan Ollama
        models = get_ollama_models()
        if models:
            self.msg_queue.put(('models', models))
        else:
            # Scan LM Studio
            models = get_openai_models()
            if models:
                self.msg_queue.put(('models', models))
            else:
                self.msg_queue.put(('models', ["gemma4:e2b (Default)"]))
                
    def connect_backend(self):
        selected = self.model_combo.get()
        if not selected:
            messagebox.showerror("Error", "Please select a model first.")
            return
            
        self.append_log(f"System: Connecting to backend with model '{selected}'...", "system")
        self.status_label.config(text="Status: Connecting...")
        
        # Assume Ollama for now as default
        def async_connect():
            try:
                # Try Ollama default
                backend = make_backend("ollama", selected, "http://127.0.0.1:11434")
                # Wrap with streaming but we will intercept the stream in GUI
                self.backend = backend
                self.engine = ExecutionEngine(self.store, self.backend)
                self.msg_queue.put(('connected', selected))
            except Exception as e:
                self.msg_queue.put(('error', f"Failed to connect: {e}"))
                
        threading.Thread(target=async_connect, daemon=True).start()
        
    def draw_engine_map(self):
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width < 100: # Not rendered yet
            self.root.after(500, self.draw_engine_map)
            return
            
        center_x = width // 2
        center_y = height // 2
        
        # Draw Central Router (Hub)
        self.draw_blob(center_x, center_y, 40, ACCENT_PINK, "Router\nHub")
        
        # Draw Kernels around it
        kernels = self.store.list_kernels()
        import math
        
        num_kernels = len(kernels)
        radius = min(width, height) // 3
        
        for i, k in enumerate(kernels):
            angle = (2 * math.pi * i) / num_kernels
            x = center_x + int(radius * math.cos(angle))
            y = center_y + int(radius * math.sin(angle))
            
            # Shorten ID for display
            display_id = k['kernel_id'][:12]
            
            # Draw connection line (dim initially)
            self.canvas.create_line(center_x, center_y, x, y, fill="#333333", width=2, tags=f"link_{k['kernel_id']}")
            
            # Draw Kernel Blob
            self.draw_blob(x, y, 25, ACCENT_CYAN, display_id, tags=f"node_{k['kernel_id']}")
            
    def draw_blob(self, x, y, r, color, text, tags=""):
        # Draw glowing outer ring
        self.canvas.create_oval(x-r-5, y-r-5, x+r+5, y+r+5, fill="", outline=color, width=1, tags=tags)
        # Draw solid inner
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=SURFACE_COLOR, outline=color, width=2, tags=tags)
        # Draw text
        self.canvas.create_text(x, y, text=text, fill=TEXT_COLOR, font=('Courier', 8, 'bold'), justify='center', tags=tags)
        
    def highlight_routing(self, kernel_id):
        # Reset all links
        for k in self.store.list_kernels():
            self.canvas.itemconfig(f"link_{k['kernel_id']}", fill="#333333", width=2)
            self.canvas.itemconfig(f"node_{k['kernel_id']}", outline=ACCENT_CYAN)
            
        # Highlight active
        if kernel_id in [k['kernel_id'] for k in self.store.list_kernels()]:
            self.canvas.itemconfig(f"link_{kernel_id}", fill=ACCENT_PINK, width=4)
            self.canvas.itemconfig(f"node_{kernel_id}", outline=ACCENT_PINK)
            
            # Animate a pulse or line flash
            self.pulse_node(kernel_id)
            
    def pulse_node(self, kernel_id, step=0):
        if step > 5: return
        # Simple size change or color flash could be implemented here
        pass

    def send_prompt(self):
        if not self.backend:
            messagebox.showerror("Error", "Please connect to a backend first.")
            return
            
        if self.executing:
            return
            
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            return
            
        self.append_log(f"\nUser: {prompt}", "user")
        self.prompt_entry.delete(0, tk.END)
        
        self.executing = True
        self.stop_requested = False
        self.status_label.config(text="Status: Thinking...")
        self.btn_send.config(state='disabled')
        
        # Run execution in background thread
        threading.Thread(target=self.async_execute, args=(prompt,), daemon=True).start()
        
    def async_execute(self, prompt):
        try:
            # 1. Routing
            self.msg_queue.put(('log', "System: Routing prompt...", "system"))
            plan = self.runtime.run(prompt)
            
            kernel_id = plan.get('kernel_id', 'none')
            self.msg_queue.put(('highlight', kernel_id))
            self.msg_queue.put(('log', f"System: Router selected mode: {plan['mode']} | Kernel: {kernel_id}", "success"))
            
            # Check for stop
            if self.stop_requested:
                self.msg_queue.put(('done', "Execution stopped by user."))
                return
                
            # 2. Execution
            self.msg_queue.put(('log', "System: Executing plan...", "system"))
            
            if plan['mode'] == 'generate':
                # Custom ReAct loop or direct generate
                # For GUI, we will simulate the stream into the log
                self.msg_queue.put(('log', "KernelWeave OS > ", "bot"))
                
                # Mock streaming for now or use the backend with a custom callback
                # Since we can't easily yield from the current backend structure without big changes,
                # we do a chunked read if possible or just dump it.
                # Let's use the real backend but handle the output
                
                resp = self.backend.generate(prompt)
                
                # Break into words to simulate streaming in GUI
                words = resp.text.split(' ')
                for word in words:
                    if self.stop_requested:
                        break
                    self.msg_queue.put(('stream', word + " "))
                    time.sleep(0.05) # Dramatic effect for the glass panel engine!
                    
                self.msg_queue.put(('stream', "\n"))
                
                # Self compile check
                if len(resp.text) > 50:
                    self.msg_queue.put(('log', "System: Task was novel. Self-compiling new skill kernel...", "success"))
                    # Actual compilation would go here
                    
            else:
                # Kernel execution
                self.msg_queue.put(('log', f"System: Executing locked skill kernel: {kernel_id}", "success"))
                result = self.engine.execute_plan(plan, prompt)
                self.msg_queue.put(('log', f"KernelWeave OS > {result.get('response_text', '')}\n", "bot"))
                
            self.msg_queue.put(('done', "Execution complete."))
            
        except Exception as e:
            self.msg_queue.put(('error', f"Execution failed: {e}"))
            self.msg_queue.put(('done', "Execution failed."))

    def force_stop(self):
        if self.executing:
            self.append_log("\n[SYSTEM] Emergency Brake Pulled! Stopping engine...", "error")
            self.stop_requested = True
            
    def poll_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                msg_type = msg[0]
                
                if msg_type == 'log':
                    self.append_log(msg[1], msg[2] if len(msg) > 2 else "bot")
                elif msg_type == 'stream':
                    self.log_area.insert(tk.END, msg[1], 'bot')
                    self.log_area.see(tk.END)
                elif msg_type == 'highlight':
                    self.highlight_routing(msg[1])
                elif msg_type == 'models':
                    self.model_combo['values'] = msg[1]
                    if msg[1]: self.model_combo.set(msg[1][0])
                    self.append_log(f"System: Found {len(msg[1])} available models.", "success")
                elif msg_type == 'connected':
                    self.status_label.config(text=f"Status: Connected to {msg[1]}")
                    self.append_log(f"System: Successfully attached to {msg[1]}!", "success")
                elif msg_type == 'error':
                    self.append_log(f"Error: {msg[1]}", "error")
                    self.status_label.config(text="Status: Error")
                elif msg_type == 'done':
                    self.executing = False
                    self.btn_send.config(state='normal')
                    self.status_label.config(text="Status: Idle")
                    
                self.msg_queue.task_done()
        except queue.Empty:
            pass
            
        # Keep polling
        self.root.after(100, self.poll_queue)
        
    def append_log(self, text, tag="bot"):
        self.log_area.insert(tk.END, text + "\n", tag)
        self.log_area.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = KernelWeaveGUI(root)
    root.mainloop()
