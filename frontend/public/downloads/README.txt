THE LANGUAGE MACHINE — LSTM Text Generation Assignment

Files:
  lstm_text_generator.py    Documented PyTorch implementation

Quick start:
  1. Install Python 3.10 or later.
  2. Run: pip install torch
  3. Run: python lstm_text_generator.py --epochs 12

The program downloads Tiny Shakespeare to shakespeare.txt, trains an LSTM,
keeps the checkpoint with the best validation loss in best_lstm.pt, and writes
three generated passages using different seed phrases to samples.txt.

Re-run generation without retraining:
  python lstm_text_generator.py --generate-only --seed "to be or not"

Run a controlled comparison for the bonus section:
  python lstm_text_generator.py --epochs 12 --sequence-length 20 --layers 2 --checkpoint baseline.pt
  python lstm_text_generator.py --epochs 12 --sequence-length 35 --layers 2 --checkpoint long-context.pt
  python lstm_text_generator.py --epochs 12 --sequence-length 20 --layers 3 --checkpoint deep.pt

Compare printed validation losses and generated samples from each run. Move
samples.txt after each run if you wish to keep all three outputs. Reuse the
same random seed, dataset slice, and epochs for a fair comparison; note that
early stopping can end runs at different epochs. Training on a CPU can take a
while; use --max-chars 100000 for a quicker preliminary check. For the full
corpus use --max-chars 0. Do not report unrun comparisons as measured results.

Dataset: Tiny Shakespeare, public-domain works of William Shakespeare
https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare
Raw text: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

Note: The website's browser preview uses corpus-based word transitions, not
neural-network inference. The real LSTM text appears in samples.txt after
training. No pretrained checkpoint or fabricated model samples are included.
