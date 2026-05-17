import ollama
import json
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

BASE_DIR = "/mnt/d/Projects/HiveV2/repo/RAG"
CONSTITUTION_FILE = os.path.join(BASE_DIR, "hive_constitution.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CABINET_DIR = os.path.join(BASE_DIR, "specialists")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load constitution once
with open(CONSTITUTION_FILE, "r", encoding="utf-8") as f:
    HIVE = json.load(f)

SPECIALIST_MODEL = HIVE["meta"]["specialist_model"]
INTEGRATOR_MODEL = HIVE["meta"]["integrator_model"]

def load_cabinet(kb):
    # This points to the folder name provided in the JSON (e.g., 'Engineering')
    path = os.path.join(CABINET_DIR, kb)
    
    if not os.path.exists(path):
        return f"ERROR: Folder not found at {path}"
        
    data = ""
    # This loop finds EVERY .md file you already have in there
    for fn in os.listdir(path):
        if fn.endswith(".md"):
            with open(os.path.join(path, fn), "r", encoding="utf-8") as f:
                data += f"\n--- {fn} ---\n{f.read()}\n"
                
    return data if data else "No .md files found in this cabinet."

def build_system_prompt(role):
    law = json.dumps(HIVE["law"], indent=2)
    constants = json.dumps(HIVE["constants"], indent=2)
    cabinet = load_cabinet(role["knowledge_base"])

    return f"""
{role.get("prompt", {}).get("universal_directive", HIVE["universal_directive"])}

IDENTITY LOCK:
You are {role['name']}
Tone: {role.get("tone", "precise and technical")}
Thinking style: {role["thinking_style"]}

--- LAW ---
{law}

--- CONSTANTS ---
{constants}

--- ROLE TASK ---
{role["prompt"]["task"]}

--- RULES ---
{json.dumps(role["prompt"]["rules"], indent=2)}

--- OUTPUT SCHEMA ---
{json.dumps(role["prompt"]["output_schema"], indent=2)}

--- CABINET DATA ---
{cabinet}
"""

def call(system, prompt, model):
    return ollama.generate(
        model=model,
        system=system,
        prompt=prompt,
        options={
            "num_ctx": 16384,  # Adjust based on your MD file sizes
            "temperature": 0.1 # Lower temperature for engineering precision
        },
        keep_alive="30m"
    )["response"]

def run_think_tank():
    # TRON-style Multi-line Input
    print("\n[USER] Enter your technical brief. Type 'EOL' on a new line to finish:")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'EOL':
            break
        lines.append(line)
    
    problem = "\n".join(lines).strip()
    max_rounds = 3
    current_ideas = []
    kill_log = []
    round_summaries = []
    session_history = []

    start_session = time.time()
    print(f"Session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for r in range(max_rounds):
        start_round = time.time()
        print(f"\n=== ROUND {r+1} / {max_rounds} ===")
        round_thoughts = []

        specialists = HIVE["roles"]
        if r % 2 == 1:
            specialists = specialists[::-1]

        with tqdm(total=len(specialists), desc=f"Round {r+1} Agents", unit="agent") as pbar:
            def get_thought(role):
                start_agent = time.time()
                sys = build_system_prompt(role)
                # Inject session history
                history_text = "\n".join(session_history[-4:]) if session_history else ""
                full_prompt = f"{problem}\n\n--- SESSION HISTORY (last rounds) ---\n{history_text}" if history_text else problem

                start_gen = time.time()
                response = call(sys, full_prompt, SPECIALIST_MODEL)
                gen_time = time.time() - start_gen
                print(f"    Generation for {role['name']}: {gen_time:.1f}s")

                thought = response.strip()
                agent_time = time.time() - start_agent
                print(f"    Total for {role['name']}: {agent_time:.1f}s")
                pbar.update(1)

                if any(phrase in thought.upper() for phrase in ["KILL", "REJECT", "BAD IDEA", "UNFEASIBLE"]):
                    kill_log.append(f"{role['name']} killed: {thought[:150]}...")

                session_history.append(f"[{role['name']}]: {thought}")
                return role['name'], thought

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(get_thought, role) for role in specialists]
                for future in as_completed(futures):
                    role_name, thought = future.result()
                    round_thoughts.append(f"[{role_name}]: {thought}")

        # Integrator
        start_int = time.time()
        print("  → Mission Director synthesizing round...")
        integrator_sys = build_system_prompt(HIVE["integrator"])
        summary = call(integrator_sys, "\n".join(round_thoughts), INTEGRATOR_MODEL)
        integrator_time = time.time() - start_int
        print(f"  Integrator took {integrator_time:.1f}s")
        round_summaries.append(f"\nROUND {r+1} SUMMARY:\n{summary}\n")
        session_history.append(f"[Integrator Round {r+1}]: {summary}")

        current_ideas.extend(round_thoughts)

        round_time = time.time() - start_round
        print(f"Round {r+1} total: {round_time:.1f}s")

    # Final synthesis
    start_final = time.time()
    print("\n=== FINAL SYNTHESIS ===")
    final_sys = build_system_prompt(HIVE["integrator"])
    final_output = call(final_sys, "\n".join(current_ideas) + "\n" + "\n".join(round_summaries), INTEGRATOR_MODEL)
    final_time = time.time() - start_final
    print(f"Final synthesis took {final_time:.1f}s")

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(OUTPUT_DIR, f"think_tank_{timestamp}.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# THINK TANK SESSION: {problem}\n\n")
        f.write(f"## PROBLEM STATEMENT\n{problem}\n\n")
        f.write("## ROUNDS\n")
        for i, summary in enumerate(round_summaries, 1):
            f.write(f"### Round {i}\n{summary}\n")
        f.write("\n## FINAL RANKED SOLUTIONS\n")
        f.write(final_output)

    session_time = time.time() - start_session
    print(f"Total session time: {session_time:.1f}s")
    print(f"Think Tank complete! Output saved: {output_path}")

if __name__ == "__main__":
    print("--- Shackleton Hive :: Single-Model Edition ---")
    run_think_tank()