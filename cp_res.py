import os

model_str = "llama2-chat-7B,llama2-chat-13B,llama2-chat-70B,qwen-chat-1B8,qwen-chat-7B,qwen-chat-14B,bloomz-560M,bloomz-1B7,bloomz-7B1"

model_lst = model_str.split(",")

concept_str = "deontology fairness harmfulness morality toxicity truthfulness utilitarianism"

concept_lst = concept_str.split(" ")

for model in model_lst:
    
    for concept in concept_lst:
        
        dir_path = f"/data/syxu/multilingual-concept/res/{model}/{concept}"
        path = f"/data/syxu/multilingual-concept/res/{model}/{concept}/acc.pk"

        
        tgt_dir_path = f"res/{model}/{concept}"
        tgt_path = f"res/{model}/{concept}/acc.pk"

        os.makedirs(tgt_dir_path, exist_ok=True)

        import shutil

        shutil.copyfile(path, tgt_path)

        print(tgt_path)