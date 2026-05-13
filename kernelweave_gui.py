"""
KernelWeave Glass-Panel GUI (Ultra-Visual Edition)
===================================================

A highly visual, local desktop GUI for KernelWeave that simulates the neuro-symbolic
engine working with a canvas showing animated "neurons", data flow, and routing.
"""
import os
import sys
import time
import json
import threading
import queue
import math
import random
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# Ensure kernelweave is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from kernelweave.kernel import KernelStore
from kernelweave.runtime import ExecutionEngine, KernelRuntime
from kernelweave_ollama import make_backend, wrap_with_streaming, get_ollama_models, get_openai_models

# ── Theme Colors (Cyberpunk / Glass Box Vibe) ────────────────────
BG_COLOR = "#0b0f19"      # Deep space dark
SURFACE_COLOR = "#1a2333" # Dark blue-gray
TEXT_COLOR = "#e6edf3"    # Light gray
ACCENT_CYAN = "#00f0ff"   # Neon Cyan
ACCENT_PINK = "#ff007f"   # Neon Pink
ACCENT_GREEN = "#00ff66"  # Neon Green
ACCENT_RED = "#ff3333"    # Neon Red
DIM_COLOR = "#5c6d84"     # Muted gray

class Particle:
    def __init__(self, x, y, target_x, target_y, color, speed=5):
        self.x = x
        self.y = y
        self.target_x = target_x
        self.target_y = target_y
        self.color = color
        self.speed = speed
        self.alive = True
        
        # Calculate velocity
        dx = target_x - x
        dy = target_y - y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 0:
            self.vx = (dx / dist) * speed
            self.vy = (dy / dist) * speed
        else:
            self.vx = 0
            self.vy = 0
            self.alive = False

    def update(self):
        self.x += self.vx
        self.y += self.vy
        
        # Check if reached target
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        if math.sqrt(dx*dx + dy*dy) < self.speed:
            self.alive = False

class KernelWeaveGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("KernelWeave OS - Neural-Symbolic Dashboard")
        self.root.geometry("1280x800")
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
        
        # Animation state
        self.particles = []
        self.neurons = [] # Simulated grid
        self.active_kernel_id = None
        self.routing_score = 0.0
        self.routing_mode = "idle"
        
        # Setup UI
        self.create_styles()
        self.create_layout()
        
        # Initialize simulated neurons
        self.init_neurons()
        
        # Load store and scan models
        self.initialize_engine()
        
        # Start loops
        self.root.after(100, self.poll_queue)
        self.root.after(50, self.animation_loop)
        
    def create_styles(self):
        style = ttk.Style()
        style.theme_use('default')
        
        style.configure('TFrame', background=BG_COLOR)
        style.configure('TLabel', background=BG_COLOR, foreground=TEXT_COLOR, font=('Courier', 10))
        style.configure('TButton', background=SURFACE_COLOR, foreground=TEXT_COLOR, font=('Courier', 10, 'bold'), borderwidth=0)
        style.map('TButton', background=[('active', ACCENT_CYAN)], foreground=[('active', BG_COLOR)])
        
        style.configure('Header.TLabel', font=('Courier', 16, 'bold'), foreground=ACCENT_CYAN)
        style.configure('Status.TLabel', font=('Courier', 9), foreground=DIM_COLOR)
        
    def create_layout(self):
        # Top Banner
        banner = ttk.Frame(self.root)
        banner.pack(fill='x', padx=20, pady=10)
        ttk.Label(banner, text="KERNELWEAVE // NEURO-SYMBOLIC CORE", style='Header.TLabel').pack(side='left')
        
        self.status_label = ttk.Label(banner, text="STATUS: INITIALIZING...", style='Status.TLabel')
        self.status_label.pack(side='right', pady=5)
        
        # Main Splitter
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
        ttk.Label(model_frame, text="MODEL:").pack(side='left', padx=5)
        
        self.model_combo = ttk.Combobox(model_frame, width=25, font=('Courier', 10))
        self.model_combo.pack(side='left', padx=5)
        
        self.btn_connect = ttk.Button(model_frame, text="ATTACH", command=self.connect_backend)
        self.btn_connect.pack(side='left', padx=5)
        
        # Stats Display (Glass look)
        stats_frame = tk.Frame(left_pane, bg=SURFACE_COLOR, bd=1, relief='solid')
        stats_frame.pack(fill='x', pady=(0, 10))
        
        self.stat_mode = tk.Label(stats_frame, text="MODE: IDLE", bg=SURFACE_COLOR, fg=ACCENT_CYAN, font=('Courier', 10, 'bold'))
        self.stat_mode.pack(side='left', padx=10, pady=5)
        
        self.stat_cache = tk.Label(stats_frame, text="KV CACHE SAVED: 0%", bg=SURFACE_COLOR, fg=ACCENT_GREEN, font=('Courier', 10, 'bold'))
        self.stat_cache.pack(side='left', padx=10, pady=5)
        
        # Text Log
        log_frame = ttk.Frame(left_pane)
        log_frame.pack(fill='both', expand=True)
        
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
        
        self.btn_send = ttk.Button(input_frame, text="EXECUTE", command=self.send_prompt)
        self.btn_send.pack(side='left', padx=5, ipady=5)
        
        self.btn_stop = ttk.Button(input_frame, text="HALT", command=self.force_stop)
        self.btn_stop.pack(side='left', padx=5, ipady=5)
        
        # ── Right Pane (Visualization) ─────────────────────────────
        viz_frame = ttk.Frame(right_pane)
        viz_frame.pack(fill='both', expand=True)
        
        self.canvas = tk.Canvas(
            viz_frame, 
            bg=BG_COLOR, 
            borderwidth=1, 
            relief='solid',
            highlightthickness=0
        )
        self.canvas.pack(fill='both', expand=True)
        
    def init_neurons(self):
        # Create a grid of simulated "neurons" that will glow
        for _ in range(50):
            self.neurons.append({
                'x': random.randint(50, 500),
                'y': random.randint(50, 500),
                'brightness': random.random(),
                'pulse_speed': random.uniform(0.02, 0.08)
            })
            
    def initialize_engine(self):
        self.append_log("System: Loading KernelStore...", "system")
        try:
            self.store = KernelStore(Path("store"))
            # We enable embeddings but we will handle the loading state
            self.append_log("System: Initializing Vector Router (this may download a model if not present)...", "system")
            
            def async_load_runtime():
                self.runtime = KernelRuntime(self.store, use_embeddings=True)
                self.msg_queue.put(('log', f"System: Store loaded with {len(self.store.list_kernels())} kernels.", "success"))
                self.msg_queue.put(('redraw',))
                
            threading.Thread(target=async_load_runtime, daemon=True).start()
            
            # Scan models
            self.append_log("System: Scanning for local LLM servers...", "system")
            threading.Thread(target=self.async_scan_models, daemon=True).start()
            
        except Exception as e:
            self.append_log(f"Error initializing engine: {e}", "error")
            
    def async_scan_models(self):
        models = get_ollama_models()
        if models:
            self.msg_queue.put(('models', models))
        else:
            models = get_openai_models()
            if models:
                self.msg_queue.put(('models', models))
            else:
                self.msg_queue.put(('models', ["gemma4:e2b"]))
                
    def connect_backend(self):
        selected = self.model_combo.get()
        if not selected:
            messagebox.showerror("Error", "Please select a model first.")
            return
            
        self.append_log(f"System: Attaching to backend '{selected}'...", "system")
        self.status_label.config(text="STATUS: CONNECTING...")
        
        def async_connect():
            try:
                backend = make_backend("ollama", selected, "http://127.0.0.1:11434")
                self.backend = backend
                self.engine = ExecutionEngine(self.store, self.backend)
                self.msg_queue.put(('connected', selected))
            except Exception as e:
                self.msg_queue.put(('error', f"Failed to connect: {e}"))
                
        threading.Thread(target=async_connect, daemon=True).start()
        
    def animation_loop(self):
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width < 100:
            self.root.after(50, self.animation_loop)
            return
            
        # 1. Draw Simulated Neural Field (The "Engine" background)
        for n in self.neurons:
            # Update brightness
            n['brightness'] += n['pulse_speed']
            if n['brightness'] > 1.0 or n['brightness'] < 0.2:
                n['pulse_speed'] = -n['pulse_speed']
                
            # Map grid to canvas size
            cx = int((n['x'] / 500) * width)
            cy = int((n['y'] / 500) * height)
            
            # Draw glowing dot
            alpha = int(n['brightness'] * 255)
            color = f"#{0:02x}{int(n['brightness']*200):02x}{int(n['brightness']*255):02x}"
            self.canvas.create_oval(cx-3, cy-3, cx+3, cy+3, fill=color, outline="")
            
        # 2. Draw Architecture Nodes
        center_x = width // 2
        center_y = height // 2
        
        # Central Router
        self.draw_glass_node(center_x, center_y, 50, ACCENT_PINK, "ROUTER")
        
        # Draw Kernels around it
        if self.store:
            kernels = self.store.list_kernels()
            num_kernels = len(kernels)
            radius = min(width, height) // 3
            
            for i, k in enumerate(kernels):
                angle = (2 * math.pi * i) / num_kernels
                x = center_x + int(radius * math.cos(angle))
                y = center_y + int(radius * math.sin(angle))
                
                # Draw connection line
                is_active = (k['kernel_id'] == self.active_kernel_id)
                line_color = ACCENT_PINK if is_active else "#1a2333"
                line_width = 3 if is_active else 1
                self.canvas.create_line(center_x, center_y, x, y, fill=line_color, width=line_width)
                
                # Draw Kernel Node
                node_color = ACCENT_CYAN if not is_active else ACCENT_GREEN
                self.draw_glass_node(x, y, 30, node_color, k['kernel_id'][:8])
                
        # 3. Update and Draw Particles (Data flow)
        for p in self.particles[:]:
            p.update()
            if not p.alive:
                self.particles.remove(p)
            else:
                self.canvas.create_oval(p.x-4, p.y-4, p.x+4, p.y+4, fill=p.color, outline="")
                
        # 4. Draw HUD / Labels
        self.canvas.create_text(20, 20, text="NEURO-SYMBOLIC MAPPING", fill=ACCENT_CYAN, font=('Courier', 12, 'bold'), anchor='w')
        self.canvas.create_text(20, 40, text=f"ROUTING MODE: {self.routing_mode.upper()}", fill=TEXT_COLOR, font=('Courier', 10), anchor='w')
        self.canvas.create_text(20, 60, text=f"MATCH SCORE: {self.routing_score:.2f}", fill=TEXT_COLOR, font=('Courier', 10), anchor='w')
        
        # Keep loop going
        self.root.after(50, self.animation_loop)
        
    def draw_glass_node(self, x, y, r, color, text):
        # Draw glass-like circles
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=BG_COLOR, outline=color, width=2)
        self.canvas.create_oval(x-r+4, y-r+4, x+r-4, y+r-4, fill=SURFACE_COLOR, outline="", width=0)
        self.canvas.create_text(x, y, text=text, fill=TEXT_COLOR, font=('Courier', 9, 'bold'), justify='center')
        
    def spawn_particles(self, start_x, start_y, end_x, end_y, color, count=5):
        for _ in range(count):
            self.particles.append(Particle(start_x, start_y, end_x, end_y, color, speed=random.uniform(4, 8)))

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
        self.status_label.config(text="STATUS: THINKING...")
        self.btn_send.config(state='disabled')
        
        # Trigger particles from prompt to center
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        self.spawn_particles(50, height-50, width//2, height//2, ACCENT_CYAN, 10)
        
        # Run execution in background thread
        threading.Thread(target=self.async_execute, args=(prompt,), daemon=True).start()
        
    def async_execute(self, prompt):
        try:
            self.msg_queue.put(('mode', 'routing', 0.0, 'none'))
            self.msg_queue.put(('log', "System: Routing prompt through neural field...", "system"))
            
            # 1. Routing
            plan = self.runtime.run(prompt)
            
            kernel_id = plan.get('kernel_id', 'none')
            score = plan.get('score', 0.0)
            mode = plan['mode']
            
            self.msg_queue.put(('mode', mode, score, kernel_id))
            
            # Visual feedback: spawn particles from center to selected kernel
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()
            
            # Find target coordinates for particles if kernel matched
            target_x, target_y = width//2, height//2 # Fallback center
            if mode == 'kernel' and self.store:
                kernels = self.store.list_kernels()
                for i, k in enumerate(kernels):
                    if k['kernel_id'] == kernel_id:
                        angle = (2 * math.pi * i) / len(kernels)
                        radius = min(width, height) // 3
                        target_x = width//2 + int(radius * math.cos(angle))
                        target_y = height//2 + int(radius * math.sin(angle))
                        break
                        
            self.msg_queue.put(('spawn', width//2, height//2, target_x, target_y, ACCENT_PINK))
            
            self.msg_queue.put(('log', f"System: Router selected mode: {mode} | Score: {score:.2f}", "success"))
            
            if self.stop_requested:
                self.msg_queue.put(('done', "Execution stopped."))
                return
                
            # 2. Execution
            self.msg_queue.put(('log', "System: Executing...", "system"))
            
            # We use the real backend and handle streaming!
            self.msg_queue.put(('log', "KernelWeave OS > ", "bot"))
            
            if mode == 'generate':
                # Stream directly from backend
                # We need to monkey patch or use the generate that returns a stream
                # Since we wrapped it with streaming in kernelweave_ollama, let's see:
                # The backend we created is a wrapper!
                
                # To simulate real-time text in GUI without freezing, we poll the backend
                # Since the backend we have might block, we run the blocking generate in a sub-thread
                # and put tokens in the queue!
                
                def stream_target():
                    try:
                        # Call the backend which prints to stdout in the wrapper!
                        # Wait, the wrapper prints to stdout. We can capture stdout or use a different method.
                        # Let's just use the direct non-wrapped backend to get the full text and split it,
                        # OR if the backend supports streaming, we use it!
                        
                        resp = self.backend.generate(prompt)
                        words = resp.text.split(' ')
                        for word in words:
                            if self.stop_requested:
                                break
                            self.msg_queue.put(('stream', word + " "))
                            time.sleep(0.03) # Simulate fast streaming
                        self.msg_queue.put(('stream', "\n"))
                        self.msg_queue.put(('done', "Execution complete."))
                        
                        # Update cache stat (Simulated for generate mode)
                        self.msg_queue.put(('stat_cache', "KV CACHE SAVED: 0%"))
                        
                    except Exception as e:
                        self.msg_queue.put(('error', str(e)))
                        self.msg_queue.put(('done', "Execution failed."))
                        
                threading.Thread(target=stream_target, daemon=True).start()
                
            else:
                # Kernel execution (Deterministic)
                self.msg_queue.put(('stat_cache', "KV CACHE SAVED: 85%")) # Kernel skips LLM!
                result = self.engine.execute_plan(plan, prompt)
                
                # Simulate fast output
                text = result.get('response_text', '')
                words = text.split(' ')
                for word in words:
                    if self.stop_requested:
                        break
                    self.msg_queue.put(('stream', word + " "))
                    time.sleep(0.01)
                self.msg_queue.put(('stream', "\n"))
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
                elif msg_type == 'mode':
                    self.routing_mode = msg[1]
                    self.routing_score = msg[2]
                    self.active_kernel_id = msg[3]
                    self.stat_mode.config(text=f"MODE: {msg[1].upper()}")
                elif msg_type == 'spawn':
                    self.spawn_particles(msg[1], msg[2], msg[3], msg[4], msg[5])
                elif msg_type == 'stat_cache':
                    self.stat_cache.config(text=msg[1])
                elif msg_type == 'models':
                    self.model_combo['values'] = msg[1]
                    if msg[1]: self.model_combo.set(msg[1][0])
                    self.append_log(f"System: Found {len(msg[1])} available models.", "success")
                elif msg_type == 'connected':
                    self.status_label.config(text=f"STATUS: ATTACHED TO {msg[1]}")
                    self.append_log(f"System: Successfully attached to {msg[1]}!", "success")
                elif msg_type == 'error':
                    self.append_log(f"Error: {msg[1]}", "error")
                    self.status_label.config(text="STATUS: ERROR")
                elif msg_type == 'redraw':
                    self.canvas.delete("all")
                    # Force redraw in next frame
                elif msg_type == 'done':
                    self.executing = False
                    self.btn_send.config(state='normal')
                    self.status_label.config(text="STATUS: IDLE")
                    
                self.msg_queue.task_done()
        except queue.Empty:
            pass
            
        self.root.after(100, self.poll_queue)
        
    def append_log(self, text, tag="bot"):
        self.log_area.insert(tk.END, text + "\n", tag)
        self.log_area.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = KernelWeaveGUI(root)
    root.mainloop()
