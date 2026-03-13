# docker exec -it mocha-bg bash
source /home/dockeruser/mochaenv/bin/activate
cd /home/dockeruser/step_cpu
export PYTHONPATH=/home/dockeruser/step_cpu:$PYTHONPATH

PROMPT_MODES=(
    "seq_len"
    "prompt"
    "gen_token"
    "prompt_gen_token"
    "seq_len_gen_token"
)


RUNS=3
SEQ_LEN=256
MAX_NEW_TOKENS=100

SHORT_PROMPT6=(
    "Why is the sky blue?"
)
MEDIUM_PROMPT196=(
    "The history of artificial intelligence began in antiquity, with myths, stories \
    and rumors of artificial beings endowed with intelligence or consciousness by \
    master craftsmen. The seeds of modern AI were planted by philosophers who \
    attempted to describe the process of human thinking as the mechanical \
    manipulation of symbols. This work culminated in the invention of the \
    programmable digital computer in the 1940s, a machine based on the abstract \
    essence of mathematical reasoning. This device and the ideas behind it inspired \
    a handful of scientists to begin seriously discussing the possibility of building \
    an electronic brain. The field of AI research was founded at a workshop held on \
    the campus of Dartmouth College during the summer of 1956. Those who attended \
    would become the leaders of AI research for decades. Many of them predicted that \
    a machine as intelligent as a human being would exist in no more than a \
    generation, and they were given millions of dollars to make this vision come \
    true. Eventually, it became obvious that commercial developers and researchers \
    had grossly underestimated the difficulty of the project.",
)
LONG_PROMPT512=(
    "The history of artificial intelligence began in antiquity, with myths, stories \
    and rumors of artificial beings endowed with intelligence or consciousness by \
    master craftsmen. The seeds of modern AI were planted by philosophers who \
    attempted to describe the process of human thinking as the mechanical \
    manipulation of symbols. This work culminated in the invention of the \
    programmable digital computer in the 1940s, a machine based on the abstract \
    essence of mathematical reasoning. This device and the ideas behind it inspired \
    a handful of scientists to begin seriously discussing the possibility of building \
    an electronic brain. The field of AI research was founded at a workshop held on \
    the campus of Dartmouth College during the summer of 1956. Those who attended \
    would become the leaders of AI research for decades. Many of them predicted that \
    a machine as intelligent as a human being would exist in no more than a \
    generation, and they were given millions of dollars to make this vision come \
    true. Eventually, it became obvious that commercial developers and researchers \
    had grossly underestimated the difficulty of the project. In the 1970s, AI was \
    subjected to critiques and financial setbacks. AI researchers had failed to \
    appreciate the difficulty of the problems they faced. Their tremendous optimism \
    had raised expectations impossibly high, and when the promised results failed to \
    materialize, funding for AI disappeared. At the same time, the connectionism \
    movement achieved little success with simple neural network architectures. In \
    the 1980s, expert systems became commercially successful and knowledge-based \
    approaches gained momentum. The Japanese Fifth Generation Computer project \
    spurred renewed investment worldwide. However, the market for specialized AI \
    hardware collapsed in 1987, beginning the second AI winter. Research continued \
    quietly through the 1990s, with advances in machine learning, intelligent \
    agents, and statistical approaches replacing earlier symbolic methods. Deep \
    learning emerged in the 2000s as a breakthrough approach, enabled by larger \
    datasets and faster hardware. Convolutional neural networks revolutionized \
    computer vision, while recurrent networks transformed natural language \
    processing. The ImageNet competition in 2012 marked a turning point when deep \
    learning dramatically outperformed traditional methods. Tech companies invested \
    billions in AI research labs. Reinforcement learning achieved superhuman \
    performance in games like Go and chess. Generative adversarial networks created \
    realistic synthetic images. Transfer learning and pre-trained language models \
    like BERT and GPT demonstrated that large neural networks trained on vast corpora \
    could be fine-tuned for diverse downstream tasks with remarkable effectiveness \
    across many benchmarks and application domains throughout the research community."
)

# PROMPT=$SHORT_PROMPT6
# PROMPT=$MEDIUM_PROMPT196
PROMPT=$LONG_PROMPT512

for PROMPT_MODE in "${PROMPT_MODES[@]}"; do
echo ">>> Starting mode: $PROMPT_MODE"

if [ "$PROMPT_MODE" == "prompt" ]; then

    for i in $(seq 1 $RUNS); do
    echo "=== Run $i / $RUNS ==="

    python3 darpa/modified/causal_language_modeling_codegen.py --mode infer \
    --prompt "$PROMPT" --cpu-only --save-csv "prompt_base_${i}.csv"

    python3 darpa/modified/causal_language_modeling_codegen.py --mode infer \
    --prompt "$PROMPT" --cpu-only --replace gpt2attn_codegen --save-csv "prompt_codegen_${i}.csv"

    python3 darpa/modified/causal_language_modeling_codegen.py --mode infer \
    --prompt "$PROMPT" --cpu-only --replace gpt2attn --save-csv "prompt_attn_${i}.csv"
    done

elif [ "$PROMPT_MODE" == "seq_len" ]; then

    for i in $(seq 1 $RUNS); do
    echo "=== Run $i / $RUNS ==="

    python3 darpa/modified/causal_language_modeling_codegen.py --mode infer \
     --cpu-only --seq-len "$SEQ_LEN" --save-csv "seq_len_base_${i}.csv"

    python3 darpa/modified/causal_language_modeling_codegen.py --mode infer \
    --cpu-only --seq-len "$SEQ_LEN" --replace gpt2attn_codegen --save-csv "seq_len_codegen_${i}.csv"

    python3 darpa/modified/causal_language_modeling_codegen.py --mode infer \
    --cpu-only --seq-len "$SEQ_LEN" --replace gpt2attn --save-csv "seq_len_attn_${i}.csv"
    done

elif [ "$PROMPT_MODE" == "gen_token" ]; then
    for i in $(seq 1 $RUNS); do
    echo "=== Run $i / $RUNS ==="

    python3 darpa/modified/causal_language_modeling_codegen.py --mode infer \
     --cpu-only --prompt "$PROMPT" --max-new-tokens "$MAX_NEW_TOKENS" --save-csv "gen_token_base_${i}.csv"

    python3 darpa/modified/causal_language_modeling_codegen.py --mode infer \
    --cpu-only --prompt "$PROMPT" --max-new-tokens "$MAX_NEW_TOKENS" --replace gpt2attn_codegen --save-csv "gen_token_codegen_${i}.csv"

    python3 darpa/modified/causal_language_modeling_codegen.py --mode infer \
    --cpu-only --prompt "$PROMPT" --max-new-tokens "$MAX_NEW_TOKENS" --replace gpt2attn --save-csv "gen_token_attn_${i}.csv"
    done
elif [ "$PROMPT_MODE" == "prompt_gen_token" ]; then
    for i in $(seq 1 $RUNS); do
    echo "=== Run $i / $RUNS ==="

    python3 darpa/modified/causal_language_modeling_codegen.py --mode infer \
    --cpu-only --prompt "$PROMPT" --max-new-tokens "$MAX_NEW_TOKENS" --save-csv "prompt_gen_token_base_${i}.csv"

    python3 darpa/modified/causal_language_modeling_codegen.py --mode infer \
    --cpu-only --prompt "$PROMPT" --max-new-tokens "$MAX_NEW_TOKENS" --replace gpt2attn_codegen --save-csv "prompt_gen_token_codegen_${i}.csv"

    python3 darpa/modified/causal_language_modeling_codegen.py --mode infer \
    --cpu-only --prompt "$PROMPT" --max-new-tokens "$MAX_NEW_TOKENS" --replace gpt2attn --save-csv "prompt_gen_token_attn_${i}.csv"
    done
elif [ "$PROMPT_MODE" == "seq_len_gen_token" ]; then
    for i in $(seq 1 $RUNS); do
    echo "=== Run $i / $RUNS ==="

    python3 darpa/modified/causal_language_modeling_codegen.py --mode infer \
    --cpu-only --seq-len "$SEQ_LEN" --max-new-tokens "$MAX_NEW_TOKENS" --save-csv "seq_len_gen_token_base_${i}.csv"

    python3 darpa/modified/causal_language_modeling_codegen.py --mode infer \
    --cpu-only --seq-len "$SEQ_LEN" --max-new-tokens "$MAX_NEW_TOKENS" --replace gpt2attn_codegen --save-csv "seq_len_gen_token_codegen_${i}.csv"

    python3 darpa/modified/causal_language_modeling_codegen.py --mode infer \
    --cpu-only --seq-len "$SEQ_LEN" --max-new-tokens "$MAX_NEW_TOKENS" --replace gpt2attn --save-csv "seq_len_gen_token_attn_${i}.csv"
    done
fi

done