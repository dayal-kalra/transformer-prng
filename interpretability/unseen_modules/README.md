# UM Interpretability

In all python scripts there are path can be changed to point to the checkpoints and places to save data and plots. One needs to generate checkpoints using scripts provided in the other folder.

All Python scripts should be executed using the format ```./plot.sh *.py 4 6 100000 2048```, where the parameters correspond to a model checkpoint generated with depth=4, $n_{\text{heads}}=6$, 100000 training steps, and $m_{\text{test}}=2048$. 

**To obtain the checkpoint used in this section, one must additionally exclude the specific $m_{\text{test}}$ values of interest from the training set.**

List of python scripts and their corresponding Figure numbers:

- `run_and_plot_pca.py` for Figure 7 and Figure 8(a1, b1, c1)
- `run_and_plot_per_digit_acc_prune.py` for Figure 8(a2, b2, c2, a3, b3, c3)
- `run_and_plot_attn_weights.py` for Figure 9(a)
- `run_cosine.py` + `plot_cosine.py` for Figure 9(b)
- `run_and_plot_patching.py` for Figure 9(c)