"""Keras model definitions used across the motor-imagery papers.

TensorFlow is imported lazily so the non-deep-learning parts of the package
work without it.
"""
from __future__ import annotations


def _keras():
    import tensorflow as tf
    return tf, tf.keras


# --------------------------------------------------------------------------- EEGNet
def build_eegnet(nb_classes: int, Chans: int, Samples: int, dropoutRate: float = 0.5,
                 kernLength: int = 64, F1: int = 8, D: int = 2, F2: int = 16):
    """EEGNet (Lawhern et al., 2018) exactly as in the original notebook.

    The configuration that produced 75.16 % is
    ``build_eegnet(2, 64, 641, dropoutRate=0.5, kernLength=80, F1=16, D=2, F2=32)``
    and has 6,466 parameters (6,306 trainable).
    """
    tf, keras = _keras()
    L = keras.layers
    from tensorflow.keras.constraints import max_norm
    inp = L.Input(shape=(Chans, Samples, 1))
    x = L.Conv2D(F1, (1, kernLength), padding="same", use_bias=False)(inp)
    x = L.BatchNormalization()(x)
    x = L.DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.0))(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("elu")(x)
    x = L.AveragePooling2D((1, 4))(x)
    x = L.Dropout(dropoutRate)(x)
    x = L.SeparableConv2D(F2, (1, 16), padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("elu")(x)
    x = L.AveragePooling2D((1, 8))(x)
    x = L.Dropout(dropoutRate)(x)
    x = L.Flatten()(x)
    x = L.Dense(nb_classes, kernel_constraint=max_norm(0.25))(x)
    out = L.Activation("softmax")(x)
    return keras.Model(inp, out, name="EEGNet")


# --------------------------------------------------------------------------- 1D CNN
def build_conv1d_cnn(n_samples: int = 320, n_channels: int = 64, n_classes: int = 2, l2: float = 1e-4):
    """Three-block Conv1D classifier of the protocol-leakage case study.

    Input (time, channels). 186,882 parameters (185,986 trainable), matching
    Table 3 of that paper.
    """
    tf, keras = _keras()
    L = keras.layers
    reg = keras.regularizers.l2(l2)
    inp = L.Input(shape=(n_samples, n_channels))
    x = L.Conv1D(64, 5, padding="same", activation="relu", kernel_regularizer=reg)(inp)
    x = L.BatchNormalization()(x)
    x = L.MaxPooling1D(2)(x)
    x = L.Conv1D(128, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = L.BatchNormalization()(x)
    x = L.MaxPooling1D(2)(x)
    x = L.Conv1D(256, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = L.BatchNormalization()(x)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dense(128, activation="relu")(x)
    x = L.Dropout(0.4)(x)
    x = L.Dense(64, activation="relu")(x)
    x = L.Dropout(0.3)(x)
    out = L.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="Conv1D_CNN")


# --------------------------------------------------------------------------- CSP head
def build_csp_head(n_features: int = 18, l2: float = 1e-4, dropout: float = 0.4):
    """Compact Conv1D head applied to the 18-dimensional filter-bank CSP vector.

    Layer sequence of the CSP+CNN paper (Table 3). Note: with a Flatten before
    the 32-unit dense layer this has 9,473 trainable parameters; see the
    repository README for the discrepancy with the figure printed in the paper.
    """
    tf, keras = _keras()
    L = keras.layers
    reg = keras.regularizers.l2(l2)
    inp = L.Input(shape=(n_features, 1))
    x = L.Conv1D(16, 3, kernel_regularizer=reg)(inp)
    x = L.BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
    x = L.Activation("relu")(x)
    x = L.Conv1D(32, 3)(x)
    x = L.BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
    x = L.Activation("relu")(x)
    x = L.MaxPooling1D(2)(x)
    x = L.Dropout(dropout)(x)
    x = L.Flatten()(x)
    x = L.Dense(32, activation="relu")(x)
    x = L.Dropout(dropout)(x)
    x = L.Dense(16, activation="relu")(x)
    out = L.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="CSP_CNN_head")


# --------------------------------------------------------------------------- 4-class CNN
def build_fourclass_cnn(keep_prob: float = 0.75, n_classes: int = 4):
    """TF2/Keras port of the original six-block four-class CNN (EEG-DL style).

    Input: 4096-vector reshaped to 64x64x1. Two concatenation skips (conv2||conv1
    -> conv3, conv5||conv4 -> conv6), LeakyReLU, channel-wise (spatial) dropout,
    a 512-unit dense layer and a 4-way softmax. ~15.0 M parameters, dominated by
    the 28,800x512 dense layer.

    ``BatchStatNorm`` reproduces the original manual batch normalisation, which
    always normalises with the statistics of the current batch (also at test
    time, where the whole test set is one batch).
    """
    tf, keras = _keras()
    L = keras.layers

    class BatchStatNorm(L.Layer):
        def build(self, shape):
            c = int(shape[-1])
            self.gamma = self.add_weight(name="scale", shape=(c,), initializer="ones")
            self.beta = self.add_weight(name="beta", shape=(c,), initializer="zeros")

        def call(self, x):
            axes = list(range(len(x.shape) - 1))
            mean, var = tf.nn.moments(x, axes=axes)
            return tf.nn.batch_normalization(x, mean, var, self.beta, self.gamma, 1e-3)

    kinit = keras.initializers.TruncatedNormal(stddev=0.1)
    binit = keras.initializers.Constant(0.1)
    drop = 1.0 - keep_prob

    def conv(f, pad="same"):
        return L.Conv2D(f, 3, padding=pad, kernel_initializer=kinit, bias_initializer=binit)

    act = lambda: L.LeakyReLU(negative_slope=0.2)
    inp = L.Input(shape=(4096,))
    x0 = L.Reshape((64, 64, 1))(inp)
    h1 = L.SpatialDropout2D(drop)(act()(conv(32)(x0)))
    h2 = act()(BatchStatNorm()(conv(32)(h1)))
    h3 = L.SpatialDropout2D(drop)(act()(conv(64)(L.Concatenate()([h2, h1]))))
    p3 = L.MaxPooling2D(2, padding="same")(h3)
    h4 = L.SpatialDropout2D(drop)(act()(BatchStatNorm()(conv(64, "valid")(p3))))
    h5 = act()(BatchStatNorm()(conv(64)(h4)))
    h6 = L.SpatialDropout2D(drop)(act()(conv(128)(L.Concatenate()([h5, h4]))))
    p6 = L.MaxPooling2D(2, padding="same")(h6)
    f = L.Flatten()(p6)
    f = L.Dense(512, kernel_initializer=kinit, bias_initializer=binit)(f)
    f = L.Dropout(drop)(act()(BatchStatNorm()(f)))
    out = L.Dense(n_classes, activation="softmax", kernel_initializer=kinit, bias_initializer=binit)(f)
    return keras.Model(inp, out, name="FourClass_ResidualCNN")


def train_fourclass_cnn(model, X_tr, y_tr_oh, X_te, y_te_oh, epochs: int = 100, batch_size: int = 256,
                        lr_start: float = 1e-4, lr_decay_epoch: int = 30, lr_decay: float = 0.5,
                        l2_weight: float = 1e-4, seed: int = 42, log_every: int = 5):
    """Training loop of the original notebook (epochs 0..N inclusive).

    Each epoch draws ``len(X_tr)//batch_size`` random mini-batches, the learning
    rate halves every 30 epochs, and the loss is cross-entropy plus
    ``l2_weight * sum(l2_loss)`` over all trainable variables. Training accuracy
    is measured on the first 200 training rows, as in the original.
    """
    tf, keras = _keras()
    import numpy as np
    rng = np.random.default_rng(seed)
    opt = keras.optimizers.Adam(lr_start)

    @tf.function
    def step(xb, yb):
        with tf.GradientTape() as tape:
            p = model(xb, training=True)
            ce = -tf.reduce_mean(tf.reduce_sum(yb * tf.math.log(tf.clip_by_value(p, 1e-10, 1.0)), axis=1))
            loss = ce + l2_weight * tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))

    def acc(X, Y):
        p = model(X, training=False).numpy()
        return float((p.argmax(1) == Y.argmax(1)).mean())

    hist = {"train_acc": [], "test_acc": []}
    lr = lr_start
    n_batch = max(1, len(X_tr) // batch_size)
    for epoch in range(epochs + 1):
        if epoch and epoch % lr_decay_epoch == 0:
            lr = max(lr * lr_decay, 1e-6)
            opt.learning_rate.assign(lr)
        for _ in range(n_batch):
            b = rng.choice(len(X_tr), size=min(batch_size, len(X_tr)), replace=False)
            step(tf.constant(X_tr[b]), tf.constant(y_tr_oh[b], dtype=tf.float32))
        hist["train_acc"].append(acc(X_tr[:200], y_tr_oh[:200]))
        hist["test_acc"].append(acc(X_te, y_te_oh))
        if log_every and (epoch % log_every == 0 or epoch == epochs):
            print(f"Epoch {epoch:3d} | train {hist['train_acc'][-1]:.4f} | test {hist['test_acc'][-1]:.4f} | lr {lr:.6f}")
    return hist
