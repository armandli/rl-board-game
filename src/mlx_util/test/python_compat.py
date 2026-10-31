"""Checks that rlbg mlx_util (C++) and Python MLX read each other's files and
compute the same results.

usage: python python_compat.py path/to/mlx_util_test [work_dir]

Mirrors src/mlx_util/test/zoo.cpp; keep the two in sync. Flows:
  forward  Python weights -> C++ load_weights -> same outputs
  export   C++ weights -> Python load_weights -> same outputs
  train    same optimizer steps from the same weights in both languages, and
           optimizer state saved by one language resumes in the other
"""

import os
import subprocess
import sys
import tempfile

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

RTOL = 1e-4
ATOL = 1e-4
failures = []


def check_close(name, a, b, rtol=RTOL, atol=ATOL):
    if a.shape != b.shape:
        failures.append(f"{name}: shape {a.shape} != {b.shape}")
    elif not mx.allclose(a.astype(mx.float32), b.astype(mx.float32), rtol=rtol, atol=atol).item():
        err = mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
        failures.append(f"{name}: max abs diff {err}")


def compare(label, expected, actual, rtol=RTOL, atol=ATOL):
    if set(expected) != set(actual):
        failures.append(f"{label}: keys differ {sorted(set(expected) ^ set(actual))}")
    for k in sorted(set(expected) & set(actual)):
        check_close(f"{label} {k}", expected[k], actual[k], rtol, atol)


def run_cpp(binary, mode, work):
    subprocess.run([binary, mode, work], check=True)


# ---- zoo, mirrors zoo.cpp ----

class Recurrent(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(6, 8)
        self.gru = nn.GRU(8, 8)
        self.lstm = nn.LSTM(8, 4)

    def __call__(self, x):
        h = self.gru(self.rnn(x))
        hidden, cell = self.lstm(h)
        return hidden + cell


class Transformers(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.TransformerEncoder(2, 16, 4, mlp_dims=32)
        self.tr = nn.Transformer(
            dims=16, num_heads=4, num_encoder_layers=1, num_decoder_layers=1,
            mlp_dims=32, norm_first=False,
        )


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 16)
        self.pe = nn.SinusoidalPositionalEncoding(16)
        self.mha = nn.MultiHeadAttention(16, 4, value_output_dims=8, bias=True)
        self.rope = nn.RoPE(4)
        self.bil = nn.Bilinear(16, 8, 5)


def run_mlp(m, x):
    m.eval()
    return {"mlp.out": m(x["mlp.x"])}


def warmup_conv2d(m, x):
    m.train()
    mx.eval(m(x["conv2d.x"]))
    mx.eval(m.parameters())


def run_conv2d(m, x):
    m.eval()
    out = {"conv2d.eval": m(x["conv2d.x"])}
    m.train()
    out["conv2d.train"] = m(x["conv2d.x"])
    out["conv2d.running_mean"] = m.layers[1].running_mean
    out["conv2d.running_var"] = m.layers[1].running_var
    return out


def run_seq(name):
    def run(m, x):
        m.eval()
        return {f"{name}.out": m(x[f"{name}.x"])}
    return run


def run_recurrent(m, x):
    m.eval()
    return {"recurrent.out": m(x["recurrent.x"])}


def run_transformer(m, x):
    m.eval()
    src, tgt = x["transformer.src"], x["transformer.tgt"]
    src_mask = nn.MultiHeadAttention.create_additive_causal_mask(src.shape[1])
    tgt_mask = nn.MultiHeadAttention.create_additive_causal_mask(tgt.shape[1])
    return {
        "transformer.enc": m.enc(src, src_mask),
        "transformer.tr": m.tr(src, tgt, None, tgt_mask, None),
    }


def run_attention(m, x):
    m.eval()
    tokens = x["attention.tokens"]
    e = m.emb(tokens) + m.pe(x["attention.pos"])
    mask = nn.MultiHeadAttention.create_additive_causal_mask(tokens.shape[1])
    h = m.mha(e, e, e, mask)
    r = m.rope(h, offset=2)
    return {"attention.bil": m.bil(e, r), "attention.tied": m.emb.as_linear(e)}


ZOO = [
    ("mlp", ".npz", ".safetensors",
     lambda: nn.Sequential(
         nn.Linear(8, 16), nn.relu, nn.Linear(16, 16), nn.GELU(), nn.Dropout(0.1),
         nn.Linear(16, 4)),
     None, run_mlp),
    ("conv2d", ".safetensors", ".npz",
     lambda: nn.Sequential(
         nn.Conv2d(3, 8, 3, 1, 1), nn.BatchNorm(8), nn.ReLU(), nn.MaxPool2d(2),
         nn.Conv2d(8, 4, (3, 3), 2, 1), nn.GroupNorm(2, 4), nn.AvgPool2d(2, 1, 1),
         nn.Upsample(2.0, "linear"), nn.ConvTranspose2d(4, 3, 3, 2, 1, 1, 1)),
     warmup_conv2d, run_conv2d),
    ("conv1d", ".npz.compressed", ".npz",
     lambda: nn.Sequential(
         nn.Conv1d(4, 6, 3, 1, 1, 1, 2), nn.ConvTranspose1d(6, 5, 3, 2),
         nn.InstanceNorm(5, affine=True), nn.MaxPool1d(2), nn.LayerNorm(5),
         nn.AvgPool1d(3, 2), nn.RMSNorm(5), nn.Conv1d(5, 4, 2, 1, 0, 2)),
     None, run_seq("conv1d")),
    ("conv3d", ".safetensors", ".npz.compressed",
     lambda: nn.Sequential(
         nn.Conv3d(2, 3, 3, 1, 1), nn.ConvTranspose3d(3, 2, 2, 2), nn.MaxPool3d(2),
         nn.AvgPool3d((2, 2, 2), (1, 1, 1)), nn.Upsample((2.0, 1.0, 1.5))),
     None, run_seq("conv3d")),
    ("recurrent", ".npz", ".safetensors", Recurrent, None, run_recurrent),
    ("transformer", ".safetensors", ".npz", Transformers, None, run_transformer),
    ("attention", ".npz", ".safetensors", Attention, None, run_attention),
]


def weights_file(base, fmt):
    return base + (".npz" if fmt == ".npz.compressed" else fmt)


def save_weights(m, base, fmt):
    if fmt == ".npz.compressed":
        mx.savez_compressed(base + ".npz", **dict(tree_flatten(m.parameters())))
    else:
        m.save_weights(base + fmt)


def functions_outputs(x):
    out = {}

    def put(k, v):
        out["fn." + k] = v

    acts = [
        nn.Sigmoid(), nn.Mish(), nn.ReLU(), nn.ReLU2(), nn.ReLU6(), nn.Softmax(),
        nn.Softplus(), nn.Softsign(), nn.SiLU(), nn.LogSoftmax(), nn.LogSigmoid(), nn.Tanh(),
        nn.Hardswish(), nn.SELU(), nn.HardTanh(), nn.Softmin(), nn.GLU(), nn.LeakyReLU(0.2),
        nn.ELU(0.5), nn.CELU(0.7), nn.Softshrink(0.3), nn.HardShrink(0.3), nn.Step(0.1),
        nn.PReLU(8), nn.GELU(), nn.GELU("precise"), nn.GELU("fast"),
    ]
    for i, act in enumerate(acts):
        put(f"act.{i}", act(x["fn.act_x"]))

    L = nn.losses
    logits, targets, probs, w = x["fn.logits"], x["fn.targets"], x["fn.probs"], x["fn.weights"]
    a, b, b01, sgn, var = x["fn.a"], x["fn.b"], x["fn.b01"], x["fn.sign"], x["fn.vars"]
    put("ce", L.cross_entropy(logits, targets))
    put("ce_ls", L.cross_entropy(logits, targets, label_smoothing=0.1, reduction="mean"))
    put("ce_p", L.cross_entropy(logits, probs, weights=w, reduction="sum"))
    put("bce", L.binary_cross_entropy(a, b01))
    put("bce_nl", L.binary_cross_entropy(mx.sigmoid(a), b01, with_logits=False, reduction="none"))
    put("l1", L.l1_loss(a, b))
    put("mse", L.mse_loss(a, b, reduction="sum"))
    put("nll", L.nll_loss(nn.log_softmax(logits), targets, reduction="mean"))
    put("gnll", L.gaussian_nll_loss(a, b, var, full=True))
    put("kl", L.kl_div_loss(nn.log_softmax(a), nn.log_softmax(b)))
    put("smooth_l1", L.smooth_l1_loss(a, b, beta=0.5))
    put("triplet", L.triplet_loss(a, b, x["fn.c"]))
    put("hinge", L.hinge_loss(a, sgn))
    put("huber", L.huber_loss(a, b, delta=0.5))
    put("log_cosh", L.log_cosh_loss(a, b))
    put("cosine", L.cosine_similarity_loss(a, b, axis=1))
    put("margin", L.margin_ranking_loss(a, b, sgn, margin=0.1))

    img = x["fn.img"]
    put("up_nearest", nn.Upsample(2.0)(img))
    put("up_nearest_frac", nn.Upsample((1.5, 2.5))(img))
    put("up_linear", nn.Upsample((2.0, 3.0), "linear")(img))
    put("up_linear_ac", nn.Upsample(2.0, "linear", align_corners=True)(img))
    put("up_linear_aa", nn.Upsample((0.5, 0.75), "linear", antialias=True)(img))
    put("up_cubic", nn.Upsample(2.0, "cubic")(img))
    put("up_cubic_ac", nn.Upsample(2.0, "cubic", align_corners=True)(img))
    put("up_cubic_aa", nn.Upsample(0.5, "cubic", antialias=True)(img))
    put("up_1d", nn.Upsample(1.5, "linear")(x["fn.seq"]))

    scores = x["fn.scores"]
    put("alibi", nn.ALiBi()(scores))
    put("alibi_offset", nn.ALiBi()(scores, offset=2, mask=x["fn.score_mask"]))
    put("alibi_slope6", nn.ALiBi.create_alibi_slope(6))
    put("causal", nn.MultiHeadAttention.create_additive_causal_mask(5))
    put("spe_cos", nn.SinusoidalPositionalEncoding(8, 1e-3, 1.0, 1.0, True, True)(x["fn.pos"]))
    return out


def make_inputs():
    mx.random.seed(0)
    n = mx.random.normal
    return {
        "mlp.x": n((4, 8)),
        "conv2d.x": n((2, 8, 8, 3)),
        "conv1d.x": n((2, 10, 4)),
        "conv3d.x": n((1, 4, 4, 4, 2)),
        "recurrent.x": n((2, 5, 6)),
        "transformer.src": n((2, 7, 16)),
        "transformer.tgt": n((2, 5, 16)),
        "attention.tokens": mx.random.randint(0, 10, (2, 6)),
        "attention.pos": mx.arange(6, dtype=mx.float32),
        "fn.act_x": n((3, 8)) * 3,
        "fn.logits": n((4, 5)),
        "fn.targets": mx.array([0, 3, 4, 1]),
        "fn.probs": mx.softmax(n((4, 5)), axis=-1),
        "fn.weights": mx.array([1.0, 0.5, 2.0, 0.0]),
        "fn.a": n((4, 6)),
        "fn.b": n((4, 6)),
        "fn.c": n((4, 6)),
        "fn.b01": (mx.random.uniform(shape=(4, 6)) > 0.5).astype(mx.float32),
        "fn.sign": mx.where(mx.random.uniform(shape=(4, 6)) > 0.5, 1.0, -1.0),
        "fn.vars": mx.random.uniform(0.1, 2.0, (4, 6)),
        "fn.img": n((1, 4, 5, 2)),
        "fn.seq": n((2, 6, 3)),
        "fn.scores": n((2, 4, 3, 5)),
        "fn.score_mask": n((3, 5)),
        "fn.pos": mx.arange(4, dtype=mx.float32) * 0.5,
        "train.x": n((16, 8)),
        "train.y": n((16, 4)),
    }


# ---- training, mirrors zoo.cpp ----

def build_train_model():
    return nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 4))


OPTIMIZERS = [
    ("sgd", lambda: optim.SGD(0.05, momentum=0.9, weight_decay=1e-3, nesterov=True), 0.0),
    ("sgd_plain", lambda: optim.SGD(0.05), 0.0),
    ("sgd_damp", lambda: optim.SGD(0.05, momentum=0.9, dampening=0.2), 0.0),
    ("rmsprop", lambda: optim.RMSprop(1e-2), 0.0),
    ("adagrad", lambda: optim.Adagrad(0.05), 0.0),
    ("adadelta", lambda: optim.AdaDelta(1.0), 0.0),
    ("adam", lambda: optim.Adam(1e-2, bias_correction=True), 0.0),
    ("adamw_cos", lambda: optim.AdamW(optim.cosine_decay(1e-2, 10), weight_decay=0.05), 0.0),
    ("adamax", lambda: optim.Adamax(1e-2), 0.0),
    ("lion", lambda: optim.Lion(1e-3, weight_decay=0.01), 0.0),
    ("adafactor", lambda: optim.Adafactor(), 0.0),
    ("adafactor_b1", lambda: optim.Adafactor(
        1e-2, beta_1=0.9, weight_decay=0.01, scale_parameter=False, relative_step=False), 0.0),
    ("muon", lambda: optim.Muon(1e-2), 0.0),
    ("sgd_joined", lambda: optim.SGD(
        optim.join_schedules([optim.linear_schedule(0.0, 0.1, 2), optim.exponential_decay(0.1, 0.9)], [2]),
        momentum=0.5), 0.0),
    ("sgd_step_decay", lambda: optim.SGD(optim.step_decay(0.1, 0.5, 2)), 0.0),
    ("multi", lambda: optim.MultiOptimizer(
        # Python's MultiOptimizer cannot merge a split by list index (e.g.
        # "layers.0."), so split biases from weights
        [optim.Adam(1e-2), optim.SGD(0.05)], [lambda k, g: k.endswith("bias")]), 0.0),
    ("adam_clip", lambda: optim.Adam(1e-2), 0.5),
]


def train_step(model, opt, clip, x):
    def loss_fn():
        return nn.losses.mse_loss(model(x["train.x"]), x["train.y"])

    _, grads = nn.value_and_grad(model, loss_fn)()
    if clip > 0:
        grads, _ = optim.clip_grad_norm(grads, clip)
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state)


def load_state(opt, file):
    opt.state = tree_unflatten(list(mx.load(file).items()))


def save_state(opt, file):
    mx.save_safetensors(file, dict(tree_flatten(opt.state)))


def params(m):
    return dict(tree_flatten(m.parameters()))


def main():
    binary = os.path.abspath(sys.argv[1])
    work = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp(prefix="mlx_util_compat_")
    os.makedirs(work, exist_ok=True)
    print(f"work dir: {work}")

    inputs = make_inputs()
    mx.savez(f"{work}/inputs.npz", **inputs)

    # forward: Python weights into C++
    mx.random.seed(1)
    expected = {}
    for name, py_fmt, _, build, warmup, run in ZOO:
        m = build()
        if warmup:
            warmup(m, inputs)
        save_weights(m, f"{work}/py_{name}", py_fmt)
        expected.update(run(m, inputs))
    expected.update(functions_outputs(inputs))
    run_cpp(binary, "forward", work)
    compare("forward", expected, mx.load(f"{work}/cpp_forward.npz"))

    # export: C++ weights into Python
    run_cpp(binary, "export", work)
    cpp_out = mx.load(f"{work}/cpp_export.safetensors")
    py_out = {}
    for name, _, cpp_fmt, build, _, run in ZOO:
        m = build()
        m.load_weights(weights_file(f"{work}/cpp_{name}", cpp_fmt))
        py_out.update(run(m, inputs))
    compare("export", cpp_out, py_out)

    # training and optimizer state in both directions
    mx.random.seed(2)
    build_train_model().save_weights(f"{work}/train_w0.safetensors")
    run_cpp(binary, "train", work)
    py_w4 = {}
    for name, build_opt, clip in OPTIMIZERS:
        m = build_train_model()
        m.load_weights(f"{work}/train_w0.safetensors")
        opt = build_opt()
        for _ in range(3):
            train_step(m, opt, clip, inputs)
        compare(f"train {name} w3", params(m), mx.load(f"{work}/cpp_{name}_w3.npz"))
        m.save_weights(f"{work}/py_{name}_w3.npz")
        save_state(opt, f"{work}/py_{name}_s3.safetensors")
        cpp_state = mx.load(f"{work}/cpp_{name}_s3.safetensors")
        py_state = dict(tree_flatten(opt.state))
        compare(f"train {name} state", py_state, cpp_state)
        train_step(m, opt, clip, inputs)
        py_w4[name] = params(m)
        compare(f"train {name} w4", py_w4[name], mx.load(f"{work}/cpp_{name}_w4.npz"))

        # Python resumes from C++ weights and optimizer state
        m2 = build_train_model()
        m2.load_weights(f"{work}/cpp_{name}_w3.npz")
        opt2 = build_opt()
        load_state(opt2, f"{work}/cpp_{name}_s3.safetensors")
        train_step(m2, opt2, clip, inputs)
        compare(f"python resumes C++ {name}", mx.load(f"{work}/cpp_{name}_w4.npz"), params(m2))

    # C++ resumes from Python weights and optimizer state
    run_cpp(binary, "resume", work)
    for name, _, _ in OPTIMIZERS:
        compare(f"C++ resumes python {name}", py_w4[name],
                mx.load(f"{work}/cpp_{name}_resume_w4.safetensors"))

    checked = len(expected) + len(cpp_out) + 5 * len(OPTIMIZERS)
    if failures:
        print(f"{len(failures)} mismatches:")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print(f"python compatibility passed ({checked} comparisons)")


if __name__ == "__main__":
    main()
