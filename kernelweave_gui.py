"""
KernelWeave Glass-Panel GUI (Rupert OS - Agentic Edition)
=========================================================

A completely rebuilt GUI for KernelWeave, renamed to Rupert.
Features:
- No visualizer (as requested).
- 4 Panels:
  1. Prompt Terminal (Left)
  2. Active Kernel Display (Right Top)
  3. Active Tool Display (Right Middle)
  4. Command Terminal (Right Bottom) - Shows commands executing in realtime!
- Strict read-only terminals.
- Concise system prompt to stop wasting tokens.
- **Agentic ReAct Loop**: Automatically executes tools and feeds results back!
- **Model Dropdown**: Lists available local models!
"""
import os
import sys
import time
import json
import threading
import queue
import re
from pathlib import Path
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk

# Ensure kernelweave is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from kernelweave.kernel import KernelStore
from kernelweave.runtime import ExecutionEngine, KernelRuntime
from kernelweave_ollama import get_ollama_models, tool_web_search, tool_run_command, tool_read_file, tool_write_file, tool_list_dir

# Map of tools
TOOLS = {
    "web_search": tool_web_search,
    "run_command": tool_run_command,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "list_dir": tool_list_dir
}

# ── Theme Colors (Electric Obsidian) ───────────────────────────
BG_COLOR = "#020508"      # Very deep obsidian blue-black
SURFACE_COLOR = "#07111e" # Dark slate blue for panels
TEXT_COLOR = "#a5c4ec"    # Tech blue text
ACCENT_CYAN = "#00f0ff"   # Cyan
ACCENT_ORANGE = "#ff5500" # Warning Orange
ACCENT_GREEN = "#00ff66"  # Success Green
DIM_COLOR = "#102a45"     # Border color

SYSTEM_PROMPT = """You are Rupert, an advanced autonomous AI operating system.
You are running on a local neuro-symbolic stack.

CRITICAL: Do not waste tokens writing explanations or bullshit. Be extremely concise.
If you need to use a tool, output the JSON tool call IMMEDIATELY. Do not explain why.
Wait for the tool execution result before continuing.

You must use tools by outputting a JSON object. For example:
```json
{
  "tool": "web_search",
  "args": {"query": "latest news"}
}
```
Available tools: `web_search`, `run_command`, `read_file`, `write_file`, `list_dir`.
"""

class RupertGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Rupert OS - KernelWeave")
        self.root.geometry("1400x850")
        self.root.configure(bg=BG_COLOR)
        
        # Backend state
        self.runtime = None
        self.store = None
        self.stop_requested = False
        self.executing = False
        self.conversation_history = []
        
        # Queue for thread communication
        self.msg_queue = queue.Queue()
        
        # Presets
        self.presets = {
            "OLLAMA (Local)": "http://127.0.0.1:11434",
            "OPENAI": "https://api.openai.com/v1",
            "GEMINI": "https://generativelanguage.googleapis.com/v1"
        }
        
        # Setup UI
        self.create_layout()
        
        # Load store and scan models
        self.initialize_engine()
        
        # Start poll loop
        self.root.after(100, self.poll_queue)
        
    def create_layout(self):
        self.main_pane = tk.Frame(self.root, bg=BG_COLOR)
        self.main_pane.pack(fill='both', expand=True, padx=20, pady=20)
        
        # ── LEFT PANEL (Prompt Terminal & Controls) ───────────────────
        self.left_panel = tk.Frame(self.main_pane, bg=BG_COLOR)
        self.left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Model Selector (Dropdown!)
        tk.Label(self.left_panel, text="// CORE MODEL", fg=ACCENT_CYAN, bg=BG_COLOR, font=('Courier', 10, 'bold')).pack(anchor='w')
        
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(self.left_panel, textvariable=self.model_var, font=('Courier', 12))
        self.model_dropdown.pack(fill='x', pady=(2, 10))
        
        # Endpoint Presets
        tk.Label(self.left_panel, text="// ENDPOINT PRESETS", fg=ACCENT_CYAN, bg=BG_COLOR, font=('Courier', 10, 'bold')).pack(anchor='w')
        preset_frame = tk.Frame(self.left_panel, bg=BG_COLOR)
        preset_frame.pack(fill='x', pady=(2, 10))
        
        for name, url in self.presets.items():
            btn = tk.Button(preset_frame, text=name.split()[0], bg=DIM_COLOR, fg=TEXT_COLOR, font=('Courier', 8, 'bold'), borderwidth=0, padx=5, pady=5, 
                            command=lambda u=url: self.set_endpoint(u))
            btn.pack(side='left', padx=(0, 5))
            
        # Endpoint URL
        tk.Label(self.left_panel, text="// ENDPOINT URL", fg=ACCENT_CYAN, bg=BG_COLOR, font=('Courier', 10, 'bold')).pack(anchor='w')
        self.url_entry = tk.Entry(self.left_panel, bg=SURFACE_COLOR, fg=TEXT_COLOR, font=('Courier', 12), borderwidth=1, relief='solid', insertbackground=TEXT_COLOR)
        self.url_entry.pack(fill='x', pady=(2, 10), ipady=5)
        self.url_entry.insert(0, "http://127.0.0.1:11434")
        
        # Prompt Terminal
        tk.Label(self.left_panel, text="// PROMPT TERMINAL", fg=ACCENT_CYAN, bg=BG_COLOR, font=('Courier', 10, 'bold')).pack(anchor='w')
        self.log_area = scrolledtext.ScrolledText(
            self.left_panel, 
            bg=SURFACE_COLOR, 
            fg=TEXT_COLOR, 
            font=('Courier', 10),
            insertbackground=TEXT_COLOR,
            wrap=tk.WORD,
            borderwidth=1,
            relief='solid',
            state='disabled'
        )
        self.log_area.pack(fill='both', expand=True, pady=(2, 10))
        
        self.log_area.tag_config('user', foreground=ACCENT_CYAN)
        self.log_area.tag_config('bot', foreground=TEXT_COLOR)
        self.log_area.tag_config('system', foreground=DIM_COLOR)
        
        # Prompt Input
        tk.Label(self.left_panel, text="// COMMAND INPUT", fg=ACCENT_CYAN, bg=BG_COLOR, font=('Courier', 10, 'bold')).pack(anchor='w')
        self.prompt_entry = tk.Entry(self.left_panel, bg=SURFACE_COLOR, fg=TEXT_COLOR, font=('Courier', 12), borderwidth=1, relief='solid', insertbackground=TEXT_COLOR)
        self.prompt_entry.pack(fill='x', pady=(2, 10), ipady=12)
        self.prompt_entry.bind("<Return>", lambda e: self.send_prompt())
        
        # Buttons
        btn_frame = tk.Frame(self.left_panel, bg=BG_COLOR)
        btn_frame.pack(fill='x')
        
        self.btn_send = tk.Button(btn_frame, text="TRANSMIT", bg=ACCENT_CYAN, fg=BG_COLOR, font=('Courier', 10, 'bold'), borderwidth=0, padx=20, pady=10, command=self.send_prompt)
        self.btn_send.pack(side='left', padx=(0, 10))
        
        self.btn_stop = tk.Button(btn_frame, text="HALT", bg=ACCENT_ORANGE, fg=TEXT_COLOR, font=('Courier', 10, 'bold'), borderwidth=0, padx=20, pady=10, command=self.force_stop)
        self.btn_stop.pack(side='left')
        
        # ── RIGHT PANEL (Status & Commands) ───────────────────────────
        self.right_panel = tk.Frame(self.main_pane, bg=BG_COLOR)
        self.right_panel.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # Active Kernel Panel
        tk.Label(self.right_panel, text="// ACTIVE KERNEL", fg=ACCENT_CYAN, bg=BG_COLOR, font=('Courier', 10, 'bold')).pack(anchor='w')
        self.kernel_label = tk.Label(self.right_panel, text="None", fg=TEXT_COLOR, bg=SURFACE_COLOR, font=('Courier', 12), anchor='w', padx=10, pady=10, relief='solid', borderwidth=1)
        self.kernel_label.pack(fill='x', pady=(2, 15))
        
        # Active Tool Panel
        tk.Label(self.right_panel, text="// ACTIVE TOOL", fg=ACCENT_CYAN, bg=BG_COLOR, font=('Courier', 10, 'bold')).pack(anchor='w')
        self.tool_label = tk.Label(self.right_panel, text="None", fg=TEXT_COLOR, bg=SURFACE_COLOR, font=('Courier', 12), anchor='w', padx=10, pady=10, relief='solid', borderwidth=1)
        self.tool_label.pack(fill='x', pady=(2, 15))
        
        # Command Terminal (Shows commands executing in realtime!)
        tk.Label(self.right_panel, text="// COMMAND TERMINAL", fg=ACCENT_CYAN, bg=BG_COLOR, font=('Courier', 10, 'bold')).pack(anchor='w')
        self.cmd_area = scrolledtext.ScrolledText(
            self.right_panel, 
            bg=SURFACE_COLOR, 
            fg=ACCENT_GREEN, 
            font=('Courier', 10),
            insertbackground=TEXT_COLOR,
            wrap=tk.WORD,
            borderwidth=1,
            relief='solid',
            state='disabled'
        )
        self.cmd_area.pack(fill='both', expand=True, pady=(2, 0))
        
    def set_endpoint(self, url):
        self.url_entry.delete(0, tk.END)
        self.url_entry.insert(0, url)
        
    def initialize_engine(self):
        self.append_log("Rupert: Initializing Core Systems...", "system")
        try:
            self.store = KernelStore(Path("store"))
            self.runtime = KernelRuntime(self.store, use_embeddings=True)
            self.append_log(f"Rupert: Store online. {len(self.store.list_kernels())} kernels loaded.", "system")
            
            # Load models into dropdown
            models = get_ollama_models()
            if models:
                self.model_dropdown['values'] = models
                if "granite4.1:8b" in models:
                    self.model_dropdown.set("granite4.1:8b")
                else:
                    self.model_dropdown.set(models[0])
            else:
                self.model_dropdown['values'] = ["granite4.1:8b"]
                self.model_dropdown.set("granite4.1:8b")
                
        except Exception as e:
            self.append_log(f"Error initializing core: {e}", "system")
            
    def send_prompt(self):
        selected = self.model_var.get().strip()
        if not selected:
            messagebox.showerror("Error", "Select model.")
            return
        if self.executing: return
        prompt = self.prompt_entry.get().strip()
        if not prompt: return
        
        self.append_log(f"\nUser: {prompt}", "user")
        self.prompt_entry.delete(0, tk.END)
        self.executing = True
        self.stop_requested = False
        
        threading.Thread(target=self.async_execute, args=(prompt, selected), daemon=True).start()
        
    def async_execute(self, prompt, selected):
        try:
            # 1. Routing
            plan = self.runtime.run(prompt)
            mode = plan['mode']
            kernel_id = plan.get('kernel_id', 'None')
            
            self.msg_queue.put(('update_kernel', kernel_id))
            
            # 2. Execution Loop (Agentic ReAct!)
            self.msg_queue.put(('log', "Rupert > ", "bot"))
            
            base_url = self.url_entry.get().strip()
            url = f"{base_url}/api/generate"
            
            history_text = "\n".join(self.conversation_history[-4:]) if self.conversation_history else ""
            current_prompt = f"{SYSTEM_PROMPT}\n\nRecent History:\n{history_text}\n\nUser: {prompt}"
            
            max_iterations = 3
            for i in range(max_iterations):
                if self.stop_requested: break
                
                body = {"model": selected, "prompt": current_prompt, "stream": True}
                req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers={"content-type": "application/json"})
                
                full_response = ""
                try:
                    import urllib.request
                    with urllib.request.urlopen(req, timeout=30) as response:
                        for line in response:
                            if self.stop_requested: break
                            if line:
                                chunk = json.loads(line.decode('utf-8'))
                                token = chunk.get("response", "")
                                full_response += token
                                self.msg_queue.put(('stream', token))
                                
                    self.msg_queue.put(('stream', "\n"))
                    
                    # Check for tool calls
                    blocks = re.findall(r'```json\s*(.*?)\s*```', full_response, re.DOTALL)
                    if not blocks:
                        # Try parsing raw JSON if no backticks
                        if full_response.strip().startswith("{") and full_response.strip().endswith("}"):
                            blocks = [full_response.strip()]
                            
                    if blocks:
                        try:
                            data = json.loads(blocks[0])
                            if "tool" in data and "args" in data:
                                tool = data["tool"]
                                args = data["args"]
                                
                                self.msg_queue.put(('update_tool', tool))
                                self.msg_queue.put(('cmd', f"Executing tool: {tool}"))
                                
                                # Execute tool
                                result = self.run_tool(tool, args)
                                self.msg_queue.put(('cmd', f"Result: {result}"))
                                
                                # Feed back to model!
                                current_prompt += f"\n\nResponse:\n{full_response}\n\nTool Result ({tool}):\n{result}\n\nPlease continue based on this result."
                                self.msg_queue.put(('log', f"Rupert [Continuing loop {i+1}] > ", "bot"))
                                continue
                        except Exception as e:
                            self.msg_queue.put(('cmd', f"Failed to parse or execute tool: {e}"))
                            
                    # If no tool called or failed, we are done
                    break
                    
                except Exception as e:
                    self.msg_queue.put(('log', f"Ollama error: {e}", "system"))
                    break
                    
            self.conversation_history.append(f"User: {prompt}")
            self.conversation_history.append(f"Assistant: {full_response}")
            self.msg_queue.put(('done',))
                
        except Exception as e:
            self.msg_queue.put(('log', f"Error: {e}", "system"))
            self.msg_queue.put(('done',))
            
    def run_tool(self, tool, args):
        try:
            if tool in TOOLS:
                # Resolve /tmp on windows
                for k, v in args.items():
                    if isinstance(v, str) and v.startswith("/tmp/"):
                        args[k] = v.replace("/tmp/", "e:/kernelweave/store/")
                        
                # Fix xdg-open on windows
                if tool == "run_command":
                    cmd = args.get("command")
                    if cmd and "xdg-open" in cmd:
                        args["command"] = cmd.replace("xdg-open", "start")
                        
                result = TOOLS[tool](**args)
                return result
            return f"Tool '{tool}' not found in registry."
        except Exception as e:
            return f"Error executing tool: {e}"
            
    def poll_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                msg_type = msg[0]
                if msg_type == 'log':
                    self.append_log(msg[1], msg[2] if len(msg) > 2 else "bot")
                elif msg_type == 'stream':
                    self.log_area.config(state='normal')
                    self.log_area.insert(tk.END, msg[1], 'bot')
                    self.log_area.config(state='disabled')
                    self.log_area.see(tk.END)
                elif msg_type == 'update_kernel':
                    self.kernel_label.config(text=msg[1])
                elif msg_type == 'update_tool':
                    self.tool_label.config(text=msg[1])
                elif msg_type == 'cmd':
                    self.cmd_area.config(state='normal')
                    self.cmd_area.insert(tk.END, msg[1] + "\n")
                    self.cmd_area.config(state='disabled')
                    self.cmd_area.see(tk.END)
                elif msg_type == 'done':
                    self.executing = False
                    self.tool_label.config(text="None")
                    
                self.msg_queue.task_done()
        except queue.Empty:
            pass
        self.root.after(100, self.poll_queue)
        
    def append_log(self, text, tag="bot"):
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, text + "\n", tag)
        self.log_area.config(state='disabled')
        self.log_area.see(tk.END)
        
    def force_stop(self):
        self.stop_requested = True
        self.append_log("\n[SYSTEM] Emergency Halt requested.", "system")

if __name__ == "__main__":
    root = tk.Tk()
    app = RupertGUI(root)
    root.mainloop()
