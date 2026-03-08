import json
import os
from datasets import load_dataset

def normalize_medqa_file(filepath):
    """Normalize a MedQA JSONL file to common schema."""
    normalized = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            options = data.get('options', {})
            
            # Convert options dict to list format with indices
            options_list = []
            for key in ['A', 'B', 'C', 'D', 'E']:
                if key in options:
                    options_list.append({
                        'idx': key,
                        'text': options[key]
                    })
            
            normalized.append({
                'question': data.get('question', ''),
                'options': options_list,
                'answer': data.get('answer', ''),
                'answer_idx': data.get('answer_idx', ''),
                'source': 'medqa',
                'meta_info': data.get('meta_info', '')
            })
    return normalized

def normalize_afrimedqa(dataset):
    """Normalize AfrimedQA dataset to common schema."""
    normalized = []
    for item in dataset:
        options_list = []
        answer_idx = None
        correct_answer = ''
        
        if item.get('answer_options'):
            # answer_options can be a JSON string or dict
            options_dict = item['answer_options']
            if isinstance(options_dict, str):
                try:
                    options_dict = json.loads(options_dict)
                except:
                    options_dict = None
            
            if options_dict and isinstance(options_dict, dict):
                # Convert option1, option2, etc. to A, B, C, etc.
                idx_map = {f'option{i}': chr(65+i-1) for i in range(1, 6)}  # option1->A, option2->B, etc.
                for opt_key, letter_idx in idx_map.items():
                    if opt_key in options_dict:
                        options_list.append({
                            'idx': letter_idx,
                            'text': options_dict[opt_key]
                        })
                
                # Find the correct answer index
                correct_answer = item.get('correct_answer', '')
                if correct_answer:
                    # Check if it's like "option1", "option2", etc. and map to text
                    if correct_answer.startswith('option'):
                        opt_nums = correct_answer.replace('option', '').split(',')
                        answer_idxs = []
                        answer_texts = []
                        for opt_num in opt_nums:
                            try:
                                opt_num = int(opt_num.strip())
                                letter_idx = chr(65+opt_num-1)  # option1->A, option2->B
                                answer_idxs.append(letter_idx)
                                answer_texts.append(options_dict.get(f'option{opt_num}', ''))
                            except:
                                pass
                        answer_idx = ','.join(answer_idxs) if len(answer_idxs) > 1 else answer_idxs[0]
                        correct_answer = ', '.join(answer_texts) if len(answer_texts) > 1 else answer_texts[0]
                    else:
                        # It's the actual answer text
                        for opt in options_list:
                            if opt['text'] == correct_answer:
                                answer_idx = opt['idx']
                                break
        
        normalized.append({
            'question': item.get('question', ''),
            'options': options_list,
            'answer': correct_answer if correct_answer else item.get('correct_answer', ''),
            'answer_idx': answer_idx,
            'source': 'afrimedqa',
            'sample_id': item.get('sample_id', ''),
            'split': item.get('split', ''),
            'gender': item.get('gender', ''),
            'country': item.get('country', ''),
            'specialty': item.get('specialty', ''),
            'tier': item.get('tier', '')
        })
    return normalized

def save_normalized(data, filepath):
    """Save normalized data to JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(item)} records to {filepath}")

# Normalize MedQA (US)
print("Processing MedQA US...")
medqa_us_files = {
    'train': 'data/raw/med-qa/questions/US/train.jsonl',
    'dev': 'data/raw/med-qa/questions/US/dev.jsonl', 
    'test': 'data/raw/med-qa/questions/US/test.jsonl'
}

for split, filepath in medqa_us_files.items():
    if os.path.exists(filepath):
        data = normalize_medqa_file(filepath)
        save_normalized(data, f'data/processed/medqa_us_{split}.jsonl')

# Normalize MedQA (Mainland)
print("Processing MedQA Mainland...")
medqa_mainland_files = {
    'train': 'data/raw/med-qa/questions/Mainland/train.jsonl',
    'dev': 'data/raw/med-qa/questions/Mainland/dev.jsonl',
    'test': 'data/raw/med-qa/questions/Mainland/test.jsonl'
}

for split, filepath in medqa_mainland_files.items():
    if os.path.exists(filepath):
        data = normalize_medqa_file(filepath)
        save_normalized(data, f'data/processed/medqa_mainland_{split}.jsonl')

# Normalize MedQA (Taiwan)
print("Processing MedQA Taiwan...")
medqa_taiwan_files = {
    'train': 'data/raw/med-qa/questions/Taiwan/train.jsonl',
    'dev': 'data/raw/med-qa/questions/Taiwan/dev.jsonl',
    'test': 'data/raw/med-qa/questions/Taiwan/test.jsonl'
}

for split, filepath in medqa_taiwan_files.items():
    if os.path.exists(filepath):
        data = normalize_medqa_file(filepath)
        save_normalized(data, f'data/processed/medqa_taiwan_{split}.jsonl')

# Normalize AfrimedQA
print("Processing AfrimedQA...")
afrimedqa = load_dataset("intronhealth/afrimedqa_v2")

for split in ['train', 'validation', 'test']:
    if split in afrimedqa:
        data = normalize_afrimedqa(afrimedqa[split])
        save_normalized(data, f'data/processed/afrimedqa_{split}.jsonl')

print("Done!")
