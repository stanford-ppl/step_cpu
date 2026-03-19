Write me a Python code that reads the csv files generated from /home/ginasohn/darpa-mocha/sweep.sh.
The generated csv files will internally look the same like this.
```csv
stage,time_s,cpu_pct,cores_used,max_new_tokens,prompt,prompt_token_len,replace
create_tokenizer,0.1377,2.3,0.6,20,"How do weather ..so how does this work?",254,gpt2mlp_fused6 gpt2attn_codegen
load_model,0.0819,0.0,0.0,20,"How do weather ..so how does this work?",254,gpt2mlp_fused6 gpt2attn_codegen
apply_replacements,41.0643,4.5,1.1,20,"How do weather ..so how does this work?",254,gpt2mlp_fused6 gpt2attn_codegen
tokenize_prompt,0.0015,0.0,0.0,20,"How do weather ..so how does this work?",254,gpt2mlp_fused6 gpt2attn_codegen
generate,0.8533,88.7,21.3,20,"How do weather ..so how does this work?",254,gpt2mlp_fused6 gpt2attn_codegen

```

Average the 'time_s' column of the 'generate' row across the runs (in the script it's currently 3 runs) in each of the four PROMPT_MODES setups.
Generate a csv file and print it in a reader-friendly format in the terminal when I run the Python file.
The format should be like below.
```
prompt_token_len,max_new_tokens,PyTorch,STeP-codegen
254,20,1.198033333,0.9883666667
254,100,3.386166667,3.437466667
620,20,2.3014,1.922833333
620,100,6.118033333,4.786333333
```

Each row corresponds to each mode and the order should align with the PROMPT_MODES order in /home/ginasohn/darpa-mocha/sweep.sh.
The PyTorch column corresponds to the csv files with the '_base' postfix and the STeP-codegen column corresponds to the csv files with the '_codegen' postfix. The numbers in these columns are the average of the 'time_s' column of the 'generate' row across the runs in each of the four PROMPT_MODES setups.