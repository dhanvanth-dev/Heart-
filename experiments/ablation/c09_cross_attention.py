# ══════════════════════════════════════════════════════════════════════════
# 17B-9. DUAL-BRANCH NETWORK: IS CROSS-ATTENTION USEFUL? (reviewer row 14)
# MCF-BiModNet (Section IV-G of the paper) reshapes each 32-d branch output to
# a sequence of length 1 before multi-head attention. With one key, softmax
# gives weight exactly 1, so attention reduces to a linear map of the clinical
# branch (W_O·W_V·h_clin + b) and the ECG "query" has no effect. We test:
#   • clinical-only MLP (what Section 14 actually trains)
#   • single MLP on all 36 features
#   • dual branch + concatenation (no attention)
#   • dual branch + cross-attention, sequence length 1 (paper architecture)
#   • dual branch + linear layer in place of attention (theoretical equivalent)
#   • dual branch + token-level cross-attention: every feature is a token, so
#     ECG-proxy tokens attend over 13 clinical tokens (softmax is non-trivial)
# 3 × 10-fold CV, identical preprocessing/SMOTE, fixed training budget.
# ══════════════════════════════════════════════════════════════════════════
print("── 17B-9 Cross-attention ablation ───────────────────────────────────")
DL_SPLITS = SPLITS[:3 * N_FOLDS]
import tensorflow as tf

class FeatureTokens(tf.keras.layers.Layer):
    """Feature tokenizer: token_j = x_j · w_j + b_j  (one learned embedding per feature)."""
    def __init__(self, dim, **kw):
        super().__init__(**kw); self.dim = dim
    def build(self, shape):
        self.w = self.add_weight(shape=(shape[-1], self.dim), initializer='glorot_uniform', name='w')
        self.b = self.add_weight(shape=(shape[-1], self.dim), initializer='zeros', name='b')
    def call(self, x):
        return x[..., None] * self.w + self.b

class DualBranchNet(BaseEstimator, ClassifierMixin):
    def __init__(self, mode='dual_xattn', n_clin=13, epochs=200, batch_size=16, lr=1e-3, seed=SEED):
        self.mode, self.n_clin, self.epochs, self.batch_size, self.lr, self.seed = \
            mode, n_clin, epochs, batch_size, lr, seed

    def _branch(self, x, L):
        x = L.Dense(64, activation='relu')(x); x = L.BatchNormalization()(x)
        x = L.Dropout(0.4)(x); return L.Dense(32, activation='relu')(x)

    def _build(self, n_feat):
        import tensorflow as tf
        L = tf.keras.layers
        tf.keras.utils.set_random_seed(self.seed)
        inp = L.Input(shape=(n_feat,))
        clin = L.Lambda(lambda t: t[:, :self.n_clin])(inp)
        ecg = L.Lambda(lambda t: t[:, self.n_clin:])(inp)
        if self.mode == 'mlp_clinical':
            h = self._branch(clin, L)
        elif self.mode == 'mlp_all':
            h = self._branch(inp, L)
        else:
            hc, he = self._branch(clin, L), self._branch(ecg, L)
            if self.mode == 'dual_concat':
                h = L.Concatenate()([hc, he])
            elif self.mode == 'dual_xattn':           # paper: sequence length 1
                q, kv = L.Reshape((1, 32))(he), L.Reshape((1, 32))(hc)
                att = L.MultiHeadAttention(num_heads=4, key_dim=8)(q, kv)
                h = L.Flatten()(L.Add()([att, q]))
            elif self.mode == 'dual_linear':          # equivalent of the above
                h = L.Add()([L.Dense(32)(hc), he])
            elif self.mode == 'dual_xattn_tokens':    # one token per feature
                ct, et = FeatureTokens(16)(clin), FeatureTokens(16)(ecg)
                att = L.MultiHeadAttention(num_heads=4, key_dim=8)(et, ct)
                h = L.GlobalAveragePooling1D()(L.LayerNormalization()(L.Add()([att, et])))
                h = L.Concatenate()([h, hc])
            else:
                raise ValueError(self.mode)
        h = L.Dense(16, activation='relu')(h)
        out = L.Dense(1, activation='sigmoid')(h)
        model = tf.keras.Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss='binary_crossentropy')
        return model

    def fit(self, X, y):
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        self.model_ = self._build(X.shape[1])
        cbs = [tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
               tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6)]
        rng = np.random.RandomState(self.seed); perm = rng.permutation(len(y))   # shuffle before val split
        self.model_.fit(X[perm].astype('float32'), y[perm].astype('float32'), validation_split=0.2,
                        epochs=self.epochs, batch_size=self.batch_size, callbacks=cbs, verbose=0)
        self.classes_ = np.array([0, 1]); return self

    def predict_proba(self, X):
        p = self.model_.predict(X.astype('float32'), verbose=0).ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

DL_VARIANTS = {
    'DL: clinical-only MLP (Section 14)':        'mlp_clinical',
    'DL: single MLP on all 36 features':         'mlp_all',
    'DL: dual branch + concatenation':           'dual_concat',
    'DL: dual branch + cross-attention (L=1, paper)': 'dual_xattn',
    'DL: dual branch + linear layer (≡ L=1 attention)': 'dual_linear',
    'DL: dual branch + token-level cross-attention':  'dual_xattn_tokens',
}
for name, mode in DL_VARIANTS.items():
    evaluate(name, make_pipe(FEAT_FULL(), clf=DualBranchNet(mode)), splits=DL_SPLITS,
             n_jobs=min(4, os.cpu_count() or 1))

ref_dl = 'DL: dual branch + cross-attention (L=1, paper)'
abl9 = summarize([REF, '− ECG-proxy block (clinical only)'] + list(DL_VARIANTS), ref_dl, title='E9_cross_attention')
display(fmt_table(abl9))

# Numerical check: attention weights with sequence length 1 are exactly 1
mha = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=8)
q = tf.random.normal((5, 1, 32)); kv = tf.random.normal((5, 1, 32))
_, w = mha(q, kv, return_attention_scores=True)
q2 = tf.random.normal((5, 1, 32))
out1, out2 = mha(q, kv), mha(q2, kv)
print(f"Attention weights with L=1: min={float(tf.reduce_min(w)):.6f}, max={float(tf.reduce_max(w)):.6f}; "
      f"output change when the ECG query is replaced: {float(tf.reduce_max(tf.abs(out1 - out2))):.2e}")
