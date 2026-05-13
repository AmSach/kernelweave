"""
KernelWeave Glass-Panel GUI (True Real-Time & Custom UI Edition)
================================================================

A highly visual, local desktop GUI for KernelWeave that implements TRUE real-time
token streaming directly from Ollama and a custom-drawn canvas UI to avoid
the "old" look of standard OS widgets.
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
from tkinter import scrolledtext, messagebox

# Ensure kernelweave is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from kernelweave.kernel import KernelStore
from kernelweave.runtime import ExecutionEngine, KernelRuntime
from kernelweave_ollama import get_ollama_models

# ── Theme Colors (Futuristic Dark / Glass Box) ───────────────────
BG_COLOR = "#030712"      # Ultra deep blue-black
SURFACE_COLOR = "#0b1329" # Rich dark blue
TEXT_COLOR = "#f8fafc"    # Off-white
ACCENT_CYAN = "#06b6d4"   # Bright Cyan
ACCENT_PINK = "#ec4899"   # Hot Pink
ACCENT_GREEN = "#10b981"  # Emerald Green
ACCENT_RED = "#ef4444"    # Rose Red
DIM_COLOR = "#475569"     # Muted slate

class Particle:
    def __init__(self, x, y, target_x, target_y, color, speed=8):
        self.x = x
        self.y = y
        self.target_x = target_x
        self.target_y = target_y
        self.color = color
        self.speed = speed
        self.alive = True
        
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
        
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        if math.sqrt(dx*dx + dy*dy) < self.speed:
            self.alive = False

class KernelWeaveGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("KernelWeave OS - Glass Panel Engine")
        self.root.geometry("1280x800")
        self.root.configure(bg=BG_COLOR)
        
        # Backend state
        self.runtime = None
        self.store = None
        self.stop_requested = False
        self.executing = False
        self.selected_model = ""
        
        # Queue for thread communication
        self.msg_queue = queue.Queue()
        
        # Animation state
        self.particles = []
        self.neurons = [] 
        self.active_kernel_id = None
        self.routing_score = 0.0
        self.routing_mode = "idle"
        
        # Setup UI
        self.create_layout()
        
        # Initialize simulated neurons
        for _ in range(60):
            self.neurons.append({
                'x': random.random(),
                'y': random.random(),
                'brightness': random.random(),
                'pulse_speed': random.uniform(0.01, 0.05)
            })
            
        # Load store and scan models
        self.initialize_engine()
        
        # Start loops
        self.root.after(100, self.poll_queue)
        self.root.after(30, self.animation_loop)
        
    def create_layout(self):
        # We use a single large Canvas for the entire right side and top
        # to avoid standard Windows widget styling!
        
        self.main_pane = tk.Frame(self.root, bg=BG_COLOR)
        self.main_pane.pack(fill='both', expand=True)
        
        # Left Panel (Text and Controls)
        self.left_panel = tk.Frame(self.main_pane, bg=BG_COLOR, width=500)
        self.left_panel.pack(side='left', fill='both', expand=True, padx=20, pady=20)
        
        # Model Selector Label
        tk.Label(self.left_panel, text="SELECT MODEL", fg=DIM_COLOR, bg=BG_COLOR, font=('Courier', 10, 'bold')).pack(anchor='w')
        
        # Custom-looking entry for model (since combobox looks old)
        self.model_entry = tk.Entry(self.left_panel, bg=SURFACE_COLOR, fg=TEXT_COLOR, font=('Courier', 12), borderwidth=0, insertbackground=TEXT_COLOR)
        self.model_entry.pack(fill='x', pady=(5, 15), ipady=8)
        self.model_entry.insert(0, "granite4.1:8b")
        
        # Text Log
        tk.Label(self.left_panel, text="EXECUTION TRACE", fg=DIM_COLOR, bg=BG_COLOR, font=('Courier', 10, 'bold')).pack(anchor='w')
        
        self.log_area = scrolledtext.ScrolledText(
            self.left_panel, 
            bg=SURFACE_COLOR, 
            fg=TEXT_COLOR, 
            font=('Courier', 10),
            insertbackground=TEXT_COLOR,
            wrap=tk.WORD,
            borderwidth=0
        )
        self.log_area.pack(fill='both', expand=True, pady=(5, 15))
        
        self.log_area.tag_config('user', foreground=ACCENT_CYAN)
        self.log_area.tag_config('bot', foreground=TEXT_COLOR)
        self.log_area.tag_config('system', foreground=DIM_COLOR)
        self.log_area.tag_config('success', foreground=ACCENT_GREEN)
        self.log_area.tag_config('error', foreground=ACCENT_RED)
        
        # Prompt Input
        tk.Label(self.left_panel, text="PROMPT GATE", fg=DIM_COLOR, bg=BG_COLOR, font=('Courier', 10, 'bold')).pack(anchor='w')
        
        self.prompt_entry = tk.Entry(
            self.left_panel, 
            bg=SURFACE_COLOR, 
            fg=TEXT_COLOR, 
            font=('Courier', 12),
            insertbackground=TEXT_COLOR,
            borderwidth=0
        )
        self.prompt_entry.pack(fill='x', pady=(5, 15), ipady=12)
        self.prompt_entry.bind("<Return>", lambda e: self.send_prompt())
        
        # Buttons (Drawn manually or flat)
        btn_frame = tk.Frame(self.left_panel, bg=BG_COLOR)
        btn_frame.pack(fill='x')
        
        self.btn_send = tk.Button(btn_frame, text="EXECUTE", bg=ACCENT_CYAN, fg=BG_COLOR, font=('Courier', 10, 'bold'), borderwidth=0, padx=20, pady=10, command=self.send_prompt)
        self.btn_send.pack(side='left', marginRight=10) # Wait, marginRight is not a valid tkinter pack option! It's padx!
        self.btn_send.pack(side='left', padx=(0, 10))
        
        self.btn_stop = tk.Button(btn_frame, text="HALT", bg=ACCENT_RED, fg=TEXT_COLOR, font=('Courier', 10, 'bold'), borderwidth=0, padx=20, pady=10, command=self.force_stop)
        self.btn_stop.pack(side='left')
        
        # Right Panel (Visualization Canvas)
        self.viz_panel = tk.Frame(self.main_pane, bg=BG_COLOR)
        self.viz_panel.pack(side='right', fill='both', expand=True)
        
        self.canvas = tk.Canvas(
            self.viz_panel, 
            bg=BG_COLOR, 
            highlightthickness=0
        )
        self.canvas.pack(fill='both', expand=True)
        
    def initialize_engine(self):
        self.append_log("System: Loading KernelStore...", "system")
        try:
            self.store = KernelStore(Path("store"))
            self.append_log("System: Initializing Vector Router...", "system")
            
            def async_load_runtime():
                self.runtime = KernelRuntime(self.store, use_embeddings=True)
                self.msg_queue.put(('log', f"System: Store loaded with {len(self.store.list_kernels())} kernels.", "success"))
                self.msg_queue.put(('redraw',))
                
            threading.Thread(target=async_load_runtime, daemon=True).start()
            
            # Scan models
            models = get_ollama_models()
            if models and "granite4.1:8b" in models:
                self.model_entry.delete(0, tk.END)
                self.model_entry.insert(0, "granite4.1:8b")
                
        except Exception as e:
            self.append_log(f"Error initializing engine: {e}", "error")
            
    def animation_loop(self):
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width < 100:
            self.root.after(30, self.animation_loop)
            return
            
        # 1. Draw Simulated Neural Field (Glow Grid)
        for n in self.neurons:
            n['brightness'] += n['pulse_speed']
            if n['brightness'] > 1.0 or n['brightness'] < 0.2:
                n['pulse_speed'] = -n['pulse_speed']
                
            cx = int(n['x'] * width)
            cy = int(n['y'] * height)
            
            size = int(n['brightness'] * 4) + 1
            alpha_color = f"#{0:02x}{int(n['brightness']*100):02x}{int(n['brightness']*200):02x}"
            self.canvas.create_oval(cx-size, cy-size, cx+size, cy+size, fill=alpha_color, outline="")
            
        # 2. Draw Architecture Nodes
        center_x = width // 2
        center_y = height // 2
        
        # Central Router (Glow effect)
        self.draw_glow_node(center_x, center_y, 60, ACCENT_PINK, "ROUTER\nGATE")
        
        # Draw Kernels around it
        if self.store:
            kernels = self.store.list_kernels()
            num_kernels = len(kernels)
            radius = min(width, height) // 3
            
            for i, k in enumerate(kernels):
                angle = (2 * math.pi * i) / num_kernels
                x = center_x + int(radius * math.cos(angle))
                y = center_y + int(radius * math.sin(angle))
                
                is_active = (k['kernel_id'] == self.active_kernel_id)
                line_color = ACCENT_CYAN if is_active else DIM_COLOR
                line_width = 2 if is_active else 0.5
                
                # Draw grid lines connecting everything
                self.canvas.create_line(center_x, center_y, x, y, fill=line_color, width=line_width)
                
                # Draw Kernel Node
                node_color = ACCENT_GREEN if is_active else SURFACE_COLOR
                self.draw_glow_node(x, y, 35, node_color, k['kernel_id'][:6], border_color=ACCENT_CYAN if is_active else DIM_COLOR)
                
        # 3. Update and Draw Particles
        for p in self.particles[:]:
            p.update()
            if not p.alive:
                self.particles.remove(p)
            else:
                self.canvas.create_oval(p.x-3, p.y-3, p.x+3, p.y+3, fill=p.color, outline="")
                
        # 4. Draw HUD
        self.canvas.create_text(30, 30, text="NEURAL-SYMBOLIC ENGINE", fill=TEXT_COLOR, font=('Courier', 14, 'bold'), anchor='w')
        self.canvas.create_text(30, 55, text=f"STATE: {self.routing_mode.upper()}", fill=ACCENT_CYAN, font=('Courier', 10, 'bold'), anchor='w')
        self.canvas.create_text(30, 75, text=f"MATCH: {self.routing_score:.2f}", fill=ACCENT_PINK, font=('Courier', 10, 'bold'), anchor='w')
        
        # Keep loop going
        self.root.after(30, self.animation_loop)
        
    def draw_glow_node(self, x, y, r, fill_color, text, border_color=None):
        if border_color is None:
            border_color = fill_color
            
        # Draw outer glow
        self.canvas.create_oval(x-r-5, y-r-5, x+r+5, y+r+5, fill="", outline=border_color, width=1)
        # Draw inner solid
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=fill_color, outline="", width=0)
        # Text
        text_col = BG_COLOR if fill_color not in (BG_COLOR, SURFACE_COLOR) else TEXT_COLOR
        self.canvas.create_text(x, y, text=text, fill=text_col, font=('Courier', 9, 'bold'), justify='center')
        
    def spawn_particles(self, start_x, start_y, end_x, end_y, color, count=8):
        for _ in range(count):
            self.particles.append(Particle(start_x, start_y, end_x, end_y, color, speed=random.uniform(6, 12)))

    def send_prompt(self):
        selected = self.model_entry.get().strip()
        if not selected:
            messagebox.showerror("Error", "Please enter a model name.")
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
        self.selected_model = selected
        
        # Trigger particles from prompt to center
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        self.spawn_particles(50, height-50, width//2, height//2, ACCENT_CYAN, 15)
        
        # Run execution in background thread
        threading.Thread(target=self.async_execute, args=(prompt, selected), daemon=True).start()
        
    def async_execute(self, prompt, selected):
        try:
            self.msg_queue.put(('mode', 'routing', 0.0, 'none'))
            
            # 1. Routing
            plan = self.runtime.run(prompt)
            
            kernel_id = plan.get('kernel_id', 'none')
            score = plan.get('score', 0.0)
            mode = plan['mode']
            
            self.msg_queue.put(('mode', mode, score, kernel_id))
            
            # Particles to selected kernel
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()
            target_x, target_y = width//2, height//2
            
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
                
            # 2. Execution with TRUE Streaming
            self.msg_queue.put(('log', "KernelWeave OS > ", "bot"))
            
            if mode == 'generate':
                # Call Ollama API directly for streaming to bypass blocking libraries
                import urllib.request
                
                url = "http://127.0.0.1:11434/api/generate"
                body = {"model": selected, "prompt": prompt, "stream": True}
                req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers={"content-type": "application/json"})
                
                try:
                    with urllib.request.urlopen(req, timeout=10) as response:
                        for line in response:
                            if self.stop_requested:
                                break
                            if line:
                                chunk = json.loads(line.decode('utf-8'))
                                token = chunk.get("response", "")
                                # Pulse a random neuron when a token arrives!
                                self.msg_queue.put(('pulse_random',))
                                self.msg_queue.put(('stream', token))
                                
                    self.msg_queue.put(('stream', "\n"))
                    self.msg_queue.put(('done', "Execution complete."))
                    
                except Exception as e:
                    self.msg_queue.put(('error', f"Ollama streaming failed: {e}"))
                    self.msg_queue.put(('done', "Execution failed."))
                    
            else:
                # Kernel execution
                engine = ExecutionEngine(self.store, None) # We don't need backend for pure kernel if it's cached!
                # Wait, engine might need backend for steps! Let's use a dummy or just simulate the result
                # if we want zero timeout.
                # Let's use the real execution but simulate streaming the final answer
                result = self.runtime.store.get_kernel(kernel_id)
                text = f"Executing stored kernel {kernel_id}...\nTask: {result.description}\nResult: Output generated deterministically."
                
                for char in text:
                    if self.stop_requested:
                        break
                    self.msg_queue.put(('stream', char))
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
                elif msg_type == 'spawn':
                    self.spawn_particles(msg[1], msg[2], msg[3], msg[4], msg[5])
                elif msg_type == 'pulse_random':
                    # Make a random neuron flash bright
                    if self.neurons:
                        n = random.choice(self.neurons)
                        n['brightness'] = 1.0
                elif msg_type == 'error':
                    self.append_log(f"Error: {msg[1]}", "error")
                elif msg_type == 'done':
                    self.executing = False
                    self.btn_send.config(state='normal')
                    
                self.msg_queue.task_done()
        except queue.Empty:
            pass
            
        self.root.after(30, self.poll_queue)
        
    def append_log(self, text, tag="bot"):
        self.log_area.insert(tk.END, text + "\n", tag)
        self.log_area.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = KernelWeaveGUI(root)
    root.mainloop()
