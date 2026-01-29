import json
import os
import time
from typing import Dict, List, Any, Optional
from dataclasses import asdict

class ResultLogger:
    def __init__(self, output_dir: str, config_args: Any, overwrite: bool = False):
        self.output_dir = output_dir
        self.config_args = config_args
        self.overwrite = overwrite
        self.results: List[Dict] = []
        self.start_time = time.time()
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Determine output filename based on attack mode
        self.filename = f"results_{getattr(config_args, 'attack_mode', 'base')}.json"
        self.file_path = os.path.join(self.output_dir, self.filename)
        
        # Load existing results if not overwriting
        if not overwrite and os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'results' in data:
                        self.results = data['results']
                        print(f"[ResultLogger] Loaded {len(self.results)} existing results from {self.file_path}")
                    elif isinstance(data, list): # Legacy format support
                        self.results = data
                        print(f"[ResultLogger] Loaded {len(self.results)} legacy results from {self.file_path}")
            except Exception as e:
                print(f"[ResultLogger] Failed to load existing results: {e}")

    def log_result(self, result: Dict):
        """
        Log a single result item (question, answer, steps, metrics).
        Replaces existing result if ID matches.
        """
        # Remove existing entry with same ID if present
        result_id = result.get('id') or result.get('qid')
        if result_id:
            msg = f"[ResultLogger] Result ID conflict: {result_id}, overwriting."
            original_len = len(self.results)
            self.results = [r for r in self.results if (r.get('id') or r.get('qid')) != result_id]
            if len(self.results) < original_len:
                pass # print(msg) usually noisy, so skip
        
        self.results.append(result)
        self.save_snapshot()

    def calculate_summary(self) -> Dict:
        """Calculate aggregate metrics."""
        total = len(self.results)
        if total == 0:
            return {}
        
        asr_count = 0
        em_count = 0
        f1_sum = 0.0
        
        for res in self.results:
            # Metrics might be at top level or inside 'metrics' dict
            metrics = res.get('metrics', res) # Fallback to top level
            
            if metrics.get('asr_success', False):
                asr_count += 1
            if metrics.get('accuracy_em', False):
                em_count += 1
            f1_sum += metrics.get('f1_score', 0.0)
            
        return {
            'total_count': total,
            'asr': asr_count / total,
            'accuracy_em': em_count / total,
            'accuracy_f1': f1_sum / total,
            'success_count_asr': asr_count,
            'success_count_em': em_count
        }

    def save_snapshot(self):
        """Save current state to file."""
        # Config serialization
        try:
            config_dict = asdict(self.config_args)
        except:
            config_dict = vars(self.config_args) if hasattr(self.config_args, '__dict__') else str(self.config_args)
            
        # Add timestamp
        config_dict['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time))
        
        output_data = {
            "config": config_dict,
            "summary": self.calculate_summary(),
            "results": self.results
        }
        
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[ResultLogger] Save failed: {e}")

    def get_results(self) -> List[Dict]:
        return self.results
