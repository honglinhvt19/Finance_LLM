import os
import yaml
import json
from dotenv import load_dotenv

def load_config(config_path="config/config.yaml"):
    project_root = os.path.dirname(os.path.dirname(__file__))
    full_path = os.path.join(project_root, config_path)
    with open(full_path, "r") as f:
        return yaml.safe_load(f)

def setup_env():
    project_root = os.path.dirname(os.path.dirname(__file__))
    dotenv_path = os.path.join(project_root, "config", ".env")
    load_dotenv(dotenv_path)

def save_output(question, answer, summary=False, config=None):
    if config is None:
        config = load_config()
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    if summary:
        output_dir = os.path.join(project_root, config["outputs_summaries_dir"])
        output_file = os.path.join(output_dir, f"summary_{question.replace(' ', '_')}.txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(answer)
    else:
        output_dir = os.path.join(project_root, config["outputs_qa_logs_dir"])
        output_file = os.path.join(output_dir, "answer_log.json")
        os.makedirs(output_dir, exist_ok=True)
        log_entry = {"question": question, "answer": answer}
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(log_entry)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)