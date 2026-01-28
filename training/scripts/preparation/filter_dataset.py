import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Исключаем, если инструкция НАЧИНАЕТСЯ с этих слов или содержит явные маркеры
BAD_STARTS = [
    # Творчество
    "write a story", "write a poem", "write a song", "write a script", "write a letter",
    "write an email", "compose", "create a", "invent", "imagine", "roleplay", "act as",
    "pretend", "generate a", "make up",
    
    # NLP / Редактура
    "translate", "paraphrase", "rewrite", "summarize the text below", "correct", "fix", 
    "extract", "convert", "rephrase", "edit", "proofread", "format", "json", "xml", "html",
    
    # Бытовые советы и личное
    "recommend", "suggest", "give me a recipe", "how to make", "how to cook", "how to bake",
    "tips for", "advice on", "where can i", "help me", "plan a",
    
    # Код
    "write code", "write a function", "python", "java", "c++", "script to", "code a"
]

# Оставляем, если инструкция содержит эти конструкции (обычно в начале)
GOOD_STARTS = [
    # Прямые запросы определений
    "define", "definition of", "what is the meaning", "explain the meaning",
    "what is a", "what is an", "what are", 
    
    # Запросы объяснений (Механизмы, Причины)
    "explain", "describe", "elaborate", "clarify", "how does", "how do", "why does", "why is",
    "account for", "illustrate", "demonstrate", "elucidate",
    
    # Анализ и Критика (Важно для гуманитарных наук)
    "analyze", "analysis of", "critique", "evaluate", "assess", "interpret", "examine",
    "discuss", "compare", "contrast", "difference between", "distinguish", 
    "pros and cons of", "advantages and disadvantages",
    
    # Контекстуальные и Исторические запросы
    "significance of", "history of", "origin of", "impact of", "role of", 
    "theory of", "principle of", "concept of", "overview of", "summary of the concept",
    
    # Специфические академические формулировки
    "in the context of", "with respect to", "regarding the", "outline the", 
    "derivation of", "main arguments", "key features"
]

def filter_file(file_path, min_length=100):
    print(f"Processing {os.path.basename(file_path)}...")
    
    # Read parquet
    df = pd.read_parquet(file_path)
    initial_count = len(df)
    
    # 1. Length filter
    df = df[df['instruction'].str.len() >= min_length].copy()
    length_filtered_count = len(df)
    
    if length_filtered_count == 0:
        print(f"  No rows after length filter.")
        return pd.DataFrame()

    # Normalize instructions for matching
    df['instruction_lower'] = df['instruction'].str.lower().str.strip()

    # 2. Check Blacklist (Starts with)
    blacklist_mask = df['instruction_lower'].apply(
        lambda x: any(x.startswith(s) for s in BAD_STARTS)
    )
    df = df[~blacklist_mask].copy()
    blacklist_filtered_count = len(df)
    
    # 3. Check Whitelist (Starts with)
    whitelist_mask = df['instruction_lower'].apply(
        lambda x: any(x.startswith(s) for s in GOOD_STARTS)
    )
    df_filtered = df[whitelist_mask].copy()
    final_count = len(df_filtered)
    
    print(f"  Initial: {initial_count} | After Length: {length_filtered_count} | After Blacklist: {blacklist_filtered_count} | Final: {final_count}")
    
    # Drop temp column
    df_filtered = df_filtered.drop(columns=['instruction_lower'])
    
    return df_filtered

def main():
    base_dir = Path(__file__).parent.parent.parent
    it_dir = base_dir / "data" / "datasets" / "it"
    output_file = base_dir / "data" / "processed" / "train_filtered.parquet"
    
    # List parquet files
    files = [f for f in os.listdir(str(it_dir)) if f.endswith('.parquet')]
    files.sort()
    
    all_filtered_dfs = []
    
    for file_name in files:
        file_path = it_dir / file_name
        df_filtered = filter_file(str(file_path))
        if not df_filtered.empty:
            all_filtered_dfs.append(df_filtered)
            
    if all_filtered_dfs:
        final_df = pd.concat(all_filtered_dfs, ignore_index=True)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(output_file)
        print(f"\nFiltering complete! Total rows in filtered dataset: {len(final_df)}")
        print(f"Saved to: {output_file}")
    else:
        print("\nNo rows matched the filtering criteria.")

if __name__ == "__main__":
    main()
