Paper implementations for VAE and Attention, complete with reports. List of remaining quirks/tests to run

Attention:
Training error decreases, does not translate well to BLEU score. Individual components of the implementation all tested extensively. Need to test recently discovered seq2seq mask out function for STOP padding in the CE loss.

VAE:
Reconstructions work well, manifolds cram data into half of the space (that half does look correct). Need to try using tanh activations.
