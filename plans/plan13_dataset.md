Can you add a flag like `seq_len` that takes in an integer to /home/ginasohn/step_cpu/darpa/modified/causal_language_modeling_codegen.py?

Relationship with the `prompt` flag when infer mode:
- when `seq_len` and `prompt` flag are present: ignroe `seq_len`
- `prompt` flag not present but `seq_len` flag present: Sample 100 questions (`load_dataset("dany0407/eli5_category", split="test[100:200]")`) in the ELI5 test set and use the one that has a similar length (when using both the `title`(question) and `selftext`(optional elaboration)).
- when both flags are not present: error