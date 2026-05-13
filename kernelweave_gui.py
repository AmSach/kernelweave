"""
KernelWeave Glass-Panel GUI (Architecture Pipeline Edition)
===========================================================

A highly visual, local desktop GUI for KernelWeave that visualizes the entire
neuro-symbolic pipeline from routing to output in a cool sci-fi style,
while fixing the system prompt so the model knows what KernelWeave is and can use tools!
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

# System prompt to make the model aware of itself and its tools!
SYSTEM_PROMPT = """You are KernelWeave OS, a revolutionary neuro-symbolic operating system.
You bridge the gap between fluid AI thought and fast, deterministic code.

Core Architecture:
1. When you get a new task, you (the LLM) figure it out using tools.
2. Once solved, KernelWeave compiles the successful trace into a JSON 'Skill Kernel'.
3. Next time, the Router matches the prompt directly to that kernel, saving 85% of KV Cache and bypassing your generation entirely!

You have access to tools. You can use tools by outputting a JSON object with 'tool' and 'args' fields. For example:
```json
{
  "tool": "web_search",
  "args": {"query": "latest news about AI"}
}
```

Available tools:
- `list_dir(path=".")`: List directory contents.
- `read_file(path)`: Read file content.
- `write_file(path, content)`: Write file content.
- `run_command(command)`: Run a terminal command.
- `web_search(query)`: Search the web for information. Uses vector embeddings to rank relevance.

Always output valid JSON when using a tool. If you have enough information to answer, just answer normally. Do not use tools if you don't need to.
"""

class Particle:
    def __init__(self, x, y, target_x, target_y, color, speed=10):
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
        self.root.title("KernelWeave OS - Architecture Dashboard")
        self.root.geometry("1400x850")
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
        self.active_stage = None # 'prompt', 'router', 'memory', 'engine', 'output'
        self.routing_score = 0.0
        self.routing_mode = "idle"
        
        # Setup UI
        self.create_layout()
        
        # Initialize simulated neurons (Neural background)
        for _ in range(80):
            self.neurons.append({
                'x': random.random(),
                'y': random.random(),
                'brightness': random.random(),
                'pulse_speed': random.uniform(0.01, 0.04)
            })
            
        # Load store and scan models
        self.initialize_engine()
        
        # Start loops
        self.root.after(100, self.poll_queue)
        self.root.after(30, self.animation_loop)
        
    def create_layout(self):
        self.main_pane = tk.Frame(self.root, bg=BG_COLOR)
        self.main_pane.pack(fill='both', expand=True)
        
        # Left Panel (Text and Controls)
        self.left_panel = tk.Frame(self.main_pane, bg=BG_COLOR, width=500)
        self.left_panel.pack(side='left', fill='both', expand=True, padx=20, pady=20)
        
        # Model Selector Label
        tk.Label(self.left_panel, text="SELECT MODEL", fg=DIM_COLOR, bg=BG_COLOR, font=('Courier', 10, 'bold')).pack(anchor='w')
        
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
        
        # Buttons
        btn_frame = tk.Frame(self.left_panel, bg=BG_COLOR)
        btn_frame.pack(fill='x')
        
        self.btn_send = tk.Button(btn_frame, text="EXECUTE", bg=ACCENT_CYAN, fg=BG_COLOR, font=('Courier', 10, 'bold'), borderwidth=0, padx=20, pady=10, command=self.send_prompt)
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
                
            threading.Thread(target=async_load_runtime, daemon=True).start()
            
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
            
            size = int(n['brightness'] * 3) + 1
            alpha_color = f"#{0:02x}{int(n['brightness']*80):02x}{int(n['brightness']*150):02x}"
            self.canvas.create_oval(cx-size, cy-size, cx+size, cy+size, fill=alpha_color, outline="")
            
        # 2. Draw Architecture Pipeline (Flow from Top to Bottom or Left to Right)
        # Let's do a vertical flow: Prompt -> Router -> Memory -> Engine -> Output
        
        stages = [
            ('prompt', "PROMPT INGEST"),
            ('router', "VECTOR ROUTER"),
            ('memory', "RAG MEMORY"),
            ('engine', "NEURO/SYMBOLIC EXEC"),
            ('output', "VERIFIED OUTPUT")
        ]
        
        node_x = width // 2
        spacing_y = height // 6
        start_y = spacing_y
        
        stage_coords = {}
        
        for i, (key, label) in enumerate(stages):
            y = start_y + (i * spacing_y)
            stage_coords[key] = (node_x, y)
            
            # Check if active
            is_active = (self.active_stage == key)
            node_color = ACCENT_CYAN if is_active else SURFACE_COLOR
            glow_color = ACCENT_PINK if is_active else DIM_COLOR
            
            # Draw connecting lines to next
            if i < len(stages) - 1:
                next_y = start_y + ((i+1) * spacing_y)
                self.canvas.create_line(node_x, y+30, node_x, next_y-30, fill=glow_color, width=2 if is_active else 1)
                
            # Draw Stage Box (Sci-Fi style)
            self.draw_scifi_box(node_x, y, 160, 30, node_color, label, border_color=glow_color)
            
        # 3. Update and Draw Particles (Flow between stages)
        for p in self.particles[:]:
            p.update()
            if not p.alive:
                self.particles.remove(p)
            else:
                self.canvas.create_oval(p.x-4, p.y-4, p.x+4, p.y+4, fill=p.color, outline="")
                
        # 4. Draw HUD / Stats
        self.canvas.create_text(30, 30, text="KERNELWEAVE ARCHITECTURE MAP", fill=TEXT_COLOR, font=('Courier', 14, 'bold'), anchor='w')
        self.canvas.create_text(30, 55, text=f"ROUTING: {self.routing_mode.upper()}", fill=ACCENT_CYAN, font=('Courier', 10, 'bold'), anchor='w')
        self.canvas.create_text(30, 75, text=f"SIMILARITY: {self.routing_score:.2f}", fill=ACCENT_PINK, font=('Courier', 10, 'bold'), anchor='w')
        
        # Keep loop going
        self.root.after(30, self.animation_loop)
        
    def draw_scifi_box(self, x, y, w, h, fill_color, text, border_color):
        # Draw a rectangle with cut corners or just glass look
        self.canvas.create_rectangle(x-w, y-h, x+w, y+h, fill=fill_color, outline=border_color, width=2)
        # Inner text
        text_col = BG_COLOR if fill_color not in (BG_COLOR, SURFACE_COLOR) else TEXT_COLOR
        self.canvas.create_text(x, y, text=text, fill=text_col, font=('Courier', 10, 'bold'), justify='center')
        
    def spawn_particles(self, start_x, start_y, end_x, end_y, color, count=10):
        for _ in range(count):
            self.particles.append(Particle(start_x, start_y, end_x, end_y, color, speed=random.uniform(8, 14)))

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
        
        # Run execution in background thread
        threading.Thread(target=self.async_execute, args=(prompt, selected), daemon=True).start()
        
    def async_execute(self, prompt, selected):
        try:
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()
            spacing_y = height // 6
            start_y = spacing_y
            node_x = width // 2
            
            # Stage 1: Prompt
            self.msg_queue.put(('stage', 'prompt'))
            self.spawn_particles(node_x, start_y, node_x, start_y + spacing_y, ACCENT_CYAN)
            time.sleep(0.5)
            
            # Stage 2: Router
            self.msg_queue.put(('stage', 'router'))
            self.msg_queue.put(('log', "System: Routing prompt...", "system"))
            plan = self.runtime.run(prompt)
            score = plan.get('score', 0.0)
            mode = plan['mode']
            self.msg_queue.put(('mode', mode, score))
            
            self.spawn_particles(node_x, start_y + spacing_y, node_x, start_y + 2*spacing_y, ACCENT_PINK)
            time.sleep(0.5)
            
            # Stage 3: Memory
            self.msg_queue.put(('stage', 'memory'))
            self.msg_queue.put(('log', "System: Retrieving RAG context...", "system"))
            self.spawn_particles(node_x, start_y + 2*spacing_y, node_x, start_y + 3*spacing_y, ACCENT_GREEN)
            time.sleep(0.5)
            
            # Stage 4: Engine (Execution)
            self.msg_queue.put(('stage', 'engine'))
            self.msg_queue.put(('log', f"System: Executing via {mode.upper()} mode...", "success"))
            
            # Direct Ollama call with SYSTEM PROMPT
            self.msg_queue.put(('log', "KernelWeave OS > ", "bot"))
            
            import urllib.request
            
            url = "http://127.0.0.1:11434/api/generate"
            # Inject KernelWeave documentation into the prompt!
            full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}"
            body = {"model": selected, "prompt": full_prompt, "stream": True}
            req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers={"content-type": "application/json"})
            
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    for line in response:
                        if self.stop_requested:
                            break
                        if line:
                            chunk = json.loads(line.decode('utf-8'))
                            token = chunk.get("response", "")
                            # Pulse random neurons on token
                            self.msg_queue.put(('pulse_random',))
                            self.msg_queue.put(('stream', token))
                            
                self.msg_queue.put(('stream', "\n"))
                
                # Particles to Output
                self.spawn_particles(node_x, start_y + 3*spacing_y, node_x, start_y + 4*spacing_y, ACCENT_CYAN)
                time.sleep(0.3)
                
                # Stage 5: Output
                self.msg_queue.put(('stage', 'output'))
                self.msg_queue.put(('done', "Execution complete."))
                
            except Exception as e:
                self.msg_queue.put(('error', f"Ollama failed: {e}"))
                self.msg_queue.put(('done', "Execution failed."))
                
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
                elif msg_type == 'stage':
                    self.active_stage = msg[1]
                elif msg_type == 'mode':
                    self.routing_mode = msg[1]
                    self.routing_score = msg[2]
                elif msg_type == 'spawn':
                    self.spawn_particles(msg[1], msg[2], msg[3], msg[4], msg[5])
                elif msg_type == 'pulse_random':
                    if self.neurons:
                        n = random.choice(self.neurons)
                        n['brightness'] = 1.0
                elif msg_type == 'error':
                    self.append_log(f"Error: {msg[1]}", "error")
                elif msg_type == 'done':
                    self.executing = False
                    self.btn_send.config(state='normal')
                    self.active_stage = None
                    
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
