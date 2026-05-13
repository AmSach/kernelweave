"""
KernelWeave Glass-Panel GUI (JARVIS HUD Edition)
================================================

A highly visual, local desktop GUI for KernelWeave that visualizes the engine
as a complex, multi-layered Jarvis-style HUD with concentric rings, meshed nodes,
and floating data points!
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

# ── Theme Colors (Jarvis / Iron Man Palette) ────────────────────
BG_COLOR = "#050b14"      # Dark tech blue
SURFACE_COLOR = "#0b1a30" # Darker blue
TEXT_COLOR = "#a5c4ec"    # Tech blue text
ACCENT_CYAN = "#00f0ff"   # Jarvis Cyan
ACCENT_ORANGE = "#ff5500" # Warning Orange
ACCENT_PINK = "#ff0055"   # Pulse Pink
ACCENT_GREEN = "#00ff66"  # Success Green
DIM_COLOR = "#2a4d70"     # Grid lines

SYSTEM_PROMPT = """You are JARVIS (KernelWeave OS), an advanced autonomous AI operating system.
You are running on a local neuro-symbolic stack. You use an LLM for reasoning and distill repetitive tasks into Skill Kernels.

You have full access to tools. You must use tools by outputting a JSON object. For example:
```json
{
  "tool": "web_search",
  "args": {"query": "latest news"}
}
```
Available tools: `web_search`, `run_command`, `read_file`, `write_file`, `list_dir`.
Be autonomous. Act like Jarvis.
"""

class Particle:
    def __init__(self, x, y, angle, speed, color, lifetime=20):
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.lifetime = lifetime
        self.alive = True

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        if self.lifetime <= 0:
            self.alive = False

class KernelWeaveGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("JARVIS CORE - KernelWeave OS")
        self.root.geometry("1400x850")
        self.root.configure(bg=BG_COLOR)
        
        # Backend state
        self.runtime = None
        self.store = None
        self.stop_requested = False
        self.executing = False
        self.selected_model = ""
        self.conversation_history = []
        
        # Queue for thread communication
        self.msg_queue = queue.Queue()
        
        # Animation state
        self.particles = []
        self.mesh_nodes = []
        self.angle_offset = 0.0
        self.pulse_scale = 1.0
        self.pulse_dir = 0.02
        self.active_kernel_id = "None"
        self.routing_score = 0.0
        self.routing_mode = "idle"
        
        # Setup UI
        self.create_layout()
        
        # Initialize complex mesh background
        for _ in range(40):
            self.mesh_nodes.append({
                'x': random.random(),
                'y': random.random(),
                'vx': random.uniform(-0.002, 0.002),
                'vy': random.uniform(-0.002, 0.002)
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
        
        # Model Selector
        tk.Label(self.left_panel, text="CORE MODEL", fg=ACCENT_CYAN, bg=BG_COLOR, font=('Courier', 10, 'bold')).pack(anchor='w')
        self.model_entry = tk.Entry(self.left_panel, bg=SURFACE_COLOR, fg=TEXT_COLOR, font=('Courier', 12), borderwidth=1, relief='solid', insertbackground=TEXT_COLOR)
        self.model_entry.pack(fill='x', pady=(5, 15), ipady=8)
        self.model_entry.insert(0, "granite4.1:8b")
        
        # Text Log
        tk.Label(self.left_panel, text="SYSTEM TRACE", fg=ACCENT_CYAN, bg=BG_COLOR, font=('Courier', 10, 'bold')).pack(anchor='w')
        self.log_area = scrolledtext.ScrolledText(
            self.left_panel, 
            bg=SURFACE_COLOR, 
            fg=TEXT_COLOR, 
            font=('Courier', 10),
            insertbackground=TEXT_COLOR,
            wrap=tk.WORD,
            borderwidth=1,
            relief='solid'
        )
        self.log_area.pack(fill='both', expand=True, pady=(5, 15))
        
        self.log_area.tag_config('user', foreground=ACCENT_CYAN)
        self.log_area.tag_config('bot', foreground=TEXT_COLOR)
        self.log_area.tag_config('system', foreground=DIM_COLOR)
        self.log_area.tag_config('success', foreground=ACCENT_GREEN)
        self.log_area.tag_config('error', foreground=ACCENT_ORANGE)
        
        # Prompt Input
        tk.Label(self.left_panel, text="VOICE / TEXT COMMAND", fg=ACCENT_CYAN, bg=BG_COLOR, font=('Courier', 10, 'bold')).pack(anchor='w')
        self.prompt_entry = tk.Entry(self.left_panel, bg=SURFACE_COLOR, fg=TEXT_COLOR, font=('Courier', 12), borderwidth=1, relief='solid', insertbackground=TEXT_COLOR)
        self.prompt_entry.pack(fill='x', pady=(5, 15), ipady=12)
        self.prompt_entry.bind("<Return>", lambda e: self.send_prompt())
        
        # Buttons
        btn_frame = tk.Frame(self.left_panel, bg=BG_COLOR)
        btn_frame.pack(fill='x')
        
        self.btn_send = tk.Button(btn_frame, text="TRANSMIT", bg=ACCENT_CYAN, fg=BG_COLOR, font=('Courier', 10, 'bold'), borderwidth=0, padx=20, pady=10, command=self.send_prompt)
        self.btn_send.pack(side='left', padx=(0, 10))
        
        self.btn_stop = tk.Button(btn_frame, text="HALT", bg=ACCENT_ORANGE, fg=TEXT_COLOR, font=('Courier', 10, 'bold'), borderwidth=0, padx=20, pady=10, command=self.force_stop)
        self.btn_stop.pack(side='left')
        
        # Right Panel (JARVIS HUD Canvas)
        self.viz_panel = tk.Frame(self.main_pane, bg=BG_COLOR)
        self.viz_panel.pack(side='right', fill='both', expand=True)
        
        self.canvas = tk.Canvas(
            self.viz_panel, 
            bg=BG_COLOR, 
            highlightthickness=0
        )
        self.canvas.pack(fill='both', expand=True)
        
    def initialize_engine(self):
        self.append_log("JARVIS: Initializing Core Systems...", "system")
        try:
            # Self-healing: Rebuild index.json to match disk!
            import glob
            dir_path = "e:/kernelweave/store/kernels"
            kernels = []
            for f in glob.glob(os.path.join(dir_path, "*.json")):
                name = os.path.basename(f)
                kernel_id = name.replace(".json", "")
                try:
                    with open(f, "r") as kf:
                        data = json.load(kf)
                    # Extract state from status object if it exists
                    status_obj = data.get("status", {})
                    state = status_obj.get("state", "candidate") if isinstance(status_obj, dict) else "candidate"
                    
                    kernels.append({
                        "kernel_id": kernel_id,
                        "name": data.get("name", "Unknown"),
                        "task_family": data.get("task_family", "Unknown"),
                        "path": f"kernels/{name}",
                        "status": state,
                        "version": data.get("version", 2)
                    })
                except Exception as e:
                    print(f"Failed to index {name}: {e}")
            
            index_path = "e:/kernelweave/store/index.json"
            try:
                with open(index_path, "r") as ifile:
                    index_data = json.load(ifile)
            except:
                index_data = {"kernels": [], "traces": []}
                
            index_data["kernels"] = kernels
            with open(index_path, "w") as ifile:
                json.dump(index_data, ifile, indent=2)
                
            self.store = KernelStore(Path("store"))
            self.runtime = KernelRuntime(self.store, use_embeddings=True)
            self.append_log(f"JARVIS: Store online. {len(self.store.list_kernels())} kernels loaded.", "success")
            
            models = get_ollama_models()
            if models and "granite4.1:8b" in models:
                self.model_entry.delete(0, tk.END)
                self.model_entry.insert(0, "granite4.1:8b")
                
        except Exception as e:
            self.append_log(f"Error initializing core: {e}", "error")
            
    def animation_loop(self):
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width < 100:
            self.root.after(30, self.animation_loop)
            return
            
        center_x = width // 2
        center_y = height // 2
        
        # Update animations
        self.angle_offset += 0.05
        self.pulse_scale += self.pulse_dir
        if self.pulse_scale > 1.2 or self.pulse_scale < 0.8:
            self.pulse_dir = -self.pulse_dir
            
        # 1. Draw Complex Mesh Background (Neural Network)
        for n in self.mesh_nodes:
            n['x'] += n['vx']
            n['y'] += n['vy']
            if n['x'] > 1.0 or n['x'] < 0: n['vx'] = -n['vx']
            if n['y'] > 1.0 or n['y'] < 0: n['vy'] = -n['vy']
            
            cx = int(n['x'] * width)
            cy = int(n['y'] * height)
            self.canvas.create_oval(cx-2, cy-2, cx+2, cy+2, fill=DIM_COLOR, outline="")
            
            # Connect close nodes
            for n2 in self.mesh_nodes:
                if n != n2:
                    dx = n['x'] - n2['x']
                    dy = n['y'] - n2['y']
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 0.15:
                        self.canvas.create_line(cx, cy, int(n2['x']*width), int(n2['y']*height), fill="#0d213a", width=1)
                        
        # 2. Draw JARVIS Concentric Rings (The HUD)
        base_r = min(width, height) // 4
        
        # Outer dashed ring
        self.draw_dashed_circle(center_x, center_y, base_r + 50, DIM_COLOR)
        # Rotating arc ring
        self.draw_arc_ring(center_x, center_y, base_r, ACCENT_CYAN, self.angle_offset)
        # Inner pulsing ring
        self.canvas.create_oval(center_x - int(base_r*0.6*self.pulse_scale), center_y - int(base_r*0.6*self.pulse_scale),
                                center_x + int(base_r*0.6*self.pulse_scale), center_y + int(base_r*0.6*self.pulse_scale),
                                fill="", outline=ACCENT_ORANGE, width=2)
                                
        # Central Core
        self.canvas.create_oval(center_x-20, center_y-20, center_x+20, center_y+20, fill=ACCENT_CYAN, outline="")
        self.canvas.create_text(center_x, center_y, text="JARVIS", fill=BG_COLOR, font=('Courier', 8, 'bold'))
        
        # 3. Draw Floating Kernel Data Points
        if self.store:
            kernels = self.store.list_kernels()
            for i, k in enumerate(kernels):
                angle = (2 * math.pi * i) / len(kernels) + self.angle_offset * 0.2
                r = base_r + 100
                x = center_x + int(r * math.cos(angle))
                y = center_y + int(r * math.sin(angle))
                
                # Draw connecting line to core
                self.canvas.create_line(center_x, center_y, x, y, fill="#0d213a", width=1)
                # Draw node
                is_active = (k['kernel_id'] == self.active_kernel_id)
                col = ACCENT_GREEN if is_active else ACCENT_CYAN
                self.canvas.create_rectangle(x-5, y-5, x+5, y+5, fill=col, outline="")
                self.canvas.create_text(x+10, y, text=k['kernel_id'][:6], fill=TEXT_COLOR, font=('Courier', 8), anchor='w')
                
        # 4. Update and Draw Particles
        for p in self.particles[:]:
            p.update()
            if not p.alive:
                self.particles.remove(p)
            else:
                self.canvas.create_oval(p.x-2, p.y-2, p.x+2, p.y+2, fill=p.color, outline="")
                
        # 5. HUD Labels
        self.canvas.create_text(30, 30, text="// KERNELWEAVE OS //", fill=ACCENT_CYAN, font=('Courier', 14, 'bold'), anchor='w')
        self.canvas.create_text(30, 55, text=f"MODE: {self.routing_mode.upper()}", fill=TEXT_COLOR, font=('Courier', 10), anchor='w')
        self.canvas.create_text(30, 75, text=f"ACTIVE KERNEL: {self.active_kernel_id[:10]}", fill=TEXT_COLOR, font=('Courier', 10), anchor='w')
        self.canvas.create_text(30, 95, text=f"SIMILARITY: {self.routing_score:.2f}", fill=TEXT_COLOR, font=('Courier', 10), anchor='w')
        
        # Decorative tech readouts
        self.canvas.create_text(width-30, 30, text="SYS_LATTICE: OK", fill=ACCENT_GREEN, font=('Courier', 10), anchor='e')
        self.canvas.create_text(width-30, 50, text="NEURAL_LOAD: 23%", fill=TEXT_COLOR, font=('Courier', 10), anchor='e')
        self.canvas.create_text(width-30, 70, text="STORE_COUNT: " + str(len(kernels) if self.store else 0), fill=TEXT_COLOR, font=('Courier', 10), anchor='e')
        
        self.root.after(30, self.animation_loop)
        
    def draw_dashed_circle(self, x, y, r, color):
        for angle in range(0, 360, 10):
            rad1 = math.radians(angle)
            rad2 = math.radians(angle + 5)
            self.canvas.create_line(x + r*math.cos(rad1), y + r*math.sin(rad1),
                                    x + r*math.cos(rad2), y + r*math.sin(rad2), fill=color, width=1)
                                    
    def draw_arc_ring(self, x, y, r, color, offset):
        for i in range(3):
            angle = offset + (i * 120)
            self.canvas.create_arc(x-r, y-r, x+r, y+r, start=angle, extent=60, style='arc', outline=color, width=3)
            
    def send_prompt(self):
        selected = self.model_entry.get().strip()
        if not selected:
            messagebox.showerror("Error", "Enter model.")
            return
        if self.executing: return
        prompt = self.prompt_entry.get().strip()
        if not prompt: return
        
        self.append_log(f"\nUser: {prompt}", "user")
        self.prompt_entry.delete(0, tk.END)
        self.executing = True
        self.stop_requested = False
        
        # Spawn burst from core
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        for _ in range(20):
            self.particles.append(Particle(width//2, height//2, random.uniform(0, 2*math.pi), random.uniform(2, 6), ACCENT_CYAN))
            
        threading.Thread(target=self.async_execute, args=(prompt, selected), daemon=True).start()
        
    def async_execute(self, prompt, selected):
        try:
            # 1. Routing
            plan = self.runtime.run(prompt)
            mode = plan['mode']
            score = plan.get('score', 0.0)
            kernel_id = plan.get('kernel_id', 'None')
            
            self.msg_queue.put(('mode', mode, score, kernel_id))
            self.msg_queue.put(('log', f"JARVIS: Routing score {score:.2f}. Mode: {mode.upper()}", "success"))
            
            # 2. Execution
            self.msg_queue.put(('log', "JARVIS > ", "bot"))
            
            import urllib.request
            url = "http://127.0.0.1:11434/api/generate"
            
            history_text = "\n".join(self.conversation_history[-6:]) if self.conversation_history else ""
            full_prompt = f"{SYSTEM_PROMPT}\n\nRecent History:\n{history_text}\n\nUser: {prompt}"
            body = {"model": selected, "prompt": full_prompt, "stream": True}
            
            req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers={"content-type": "application/json"})
            
            full_response = ""
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    for line in response:
                        if self.stop_requested: break
                        if line:
                            chunk = json.loads(line.decode('utf-8'))
                            token = chunk.get("response", "")
                            full_response += token
                            self.msg_queue.put(('stream', token))
                            
                self.msg_queue.put(('stream', "\n"))
                self.conversation_history.append(f"User: {prompt}")
                self.conversation_history.append(f"Assistant: {full_response}")
                self.msg_queue.put(('done',))
                
            except Exception as e:
                self.msg_queue.put(('error', f"Ollama error: {e}"))
                self.msg_queue.put(('done',))
                
        except Exception as e:
            self.msg_queue.put(('error', str(e)))
            self.msg_queue.put(('done',))
            
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
                elif msg_type == 'error':
                    self.append_log(f"Error: {msg[1]}", "error")
                elif msg_type == 'done':
                    self.executing = False
                    
                self.msg_queue.task_done()
        except queue.Empty:
            pass
        self.root.after(30, self.poll_queue)
        
    def append_log(self, text, tag="bot"):
        self.log_area.insert(tk.END, text + "\n", tag)
        self.log_area.see(tk.END)
        
    def force_stop(self):
        self.stop_requested = True
        self.append_log("\n[SYSTEM] Emergency Halt requested.", "error")

if __name__ == "__main__":
    root = tk.Tk()
    app = KernelWeaveGUI(root)
    root.mainloop()
