# %%
from pathlib import Path

import altair as alt
import numpy as np
import polars as pl
import torch
from bokeh.io import output_notebook
from bokeh.plotting import figure, gridplot, show
from bokeh.resources import INLINE

# %%
from IPython.display import HTML, display

# %%
from tqdm.notebook import tqdm

# %%
from nn_from_scratch.dataloader import NaiveDataLoader
from nn_from_scratch.datasets import MNIST
from nn_from_scratch.inference import run_validation
from nn_from_scratch.layer import Softmax
from nn_from_scratch.loss import CELoss
from nn_from_scratch.model import CNNModel, load, save
from nn_from_scratch.optim import SGD, Adam

# %%
display(HTML("<style>.container { width:100% !important; }</style>"))

# %%
output_notebook(INLINE)

# %%
seed = 2025
seed_generator = torch.Generator().manual_seed(seed)


# %%
def preload(model, optimizer, model_outp, epoch: str):
    # epoch is double digit string, starting from 00
    checkpoint = f"{model_outp}.{epoch}.pt"
    # if not checkpoint.exists():
    #     raise FileNotFoundError("Failed to find model file to preload.")
    state = load(checkpoint)
    starting_epoch = state.get("epoch") + 1
    model.load_state_dict(state.get("model_state_dict"))
    optimizer.load_state_dict(state.get("optimizer_state_dict"))
    return model, optimizer, starting_epoch


# %%
def train(train, val, model, optimizer, loss, epoch_range, model_outp):
    assert len(epoch_range) == 2
    epoch_start, epoch_end = epoch_range
    assert epoch_start < epoch_end
    for e in range(epoch_start, epoch_end):
        tot_epoch_error = 0
        n_train = 0
        batch_iterator = tqdm(train_loader, desc=f"Processing epoch: {e:02d}")
        for batch in batch_iterator:
            batch_x = np.stack([train_loader.ds[idx][0] for idx in batch])
            batch_y = np.stack([train_loader.ds[idx][1] for idx in batch])
            model.train()
            optimizer.zero_grad()
            output = model(batch_x)
            batch_y_true_idx = batch_y.argmax(axis=1)
            loss = loss_fn.loss(batch_y_true_idx, output)
            grad = loss_fn.loss_prime(batch_y_true_idx, output)
            model.backward(grad)
            optimizer.step()
            assert isinstance(grad, np.ndarray)
            batch_tot_loss = np.sum(loss, axis=-1)
            tot_epoch_error += np.sum(batch_tot_loss)
            ave_batch_loss = np.mean(batch_tot_loss)
            n_train += len(batch_x)
            batch_iterator.set_postfix({"loss": f"{ave_batch_loss:.6f}"})
        avg_epoch_error = tot_epoch_error / n_train
        accuracy = run_validation(val_loader, model)
        batch_iterator.write(
            f"ave_epoch_err={e+1}/{epoch_end} "
            f"avg_epoch_error={avg_epoch_error:.6f} accuracy={accuracy:.6f}"
        )
        model_fspath = f"{model_outp}.{e:02d}.pt"
        save(
            {
                "epoch": e,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            model_fspath,
        )


# %%
mnist_data = MNIST("./mnist")
batch_size = 128
train_dataset, val_dataset = torch.utils.data.random_split(
    mnist_data.data["train"], [0.9, 0.1], generator=seed_generator
)
train_loader = NaiveDataLoader(train_dataset, batch_size, shuffle=True)
val_loader = NaiveDataLoader(val_dataset, batch_size)
test_loader = NaiveDataLoader(mnist_data.data["test"], batch_size)

# %%
input_shape = mnist_data.input_shape
output_shape = mnist_data.n_classes
print(f"{input_shape=}")
print(f"{output_shape=}")

# %%
epochs = 20
start_epoch = 0
lr = 0.001

# %%
model = CNNModel(
    input_shape,
    output_shape,
    ks=[3],
    depths=[3],
    paddings=[0],
    fc_features=[128],
)
loss_fn = CELoss()
optimizer = Adam(model.parameters(), lr, betas=(0.9, 0.99))

# %%
outdir = Path("./model")
if not outdir.exists():
    outdir.mkdir(parents=True, exist_ok=True)
model_outp = f"{model.name}.mnist"
model_outp = str(outdir / model_outp)
print(model_outp)

# %%
model, optimizer, start_epoch = preload(model, optimizer, model_outp, "19")
print(start_epoch)

# %%
train(
    train_loader,
    val_loader,
    model,
    optimizer,
    loss_fn,
    (start_epoch, epochs),
    model_outp,
)

# %%
accuracy = run_validation(test_loader, model)
print(f"Test accuracy: {accuracy:.6f}")

# %%
softmax = Softmax()
model.eval()
err_image = np.zeros_like(input_shape)
err_images = []
err_preds = []
err_truths = []
for batch in test_loader:
    batch_x = np.stack([test_loader.ds[idx][0] for idx in batch])
    batch_y = np.stack([test_loader.ds[idx][1] for idx in batch])
    output = model(batch_x)
    probs = softmax.forward(output)
    preds = np.argmax(probs, axis=-1)
    y_true = np.argmax(batch_y, axis=1)
    err_idxs = np.where(y_true != preds)[0]
    err_images += [batch_x[i] for i in err_idxs]
    err_preds += [preds[i] for i in err_idxs]
    err_truths += [y_true[i] for i in err_idxs]


# %%
err_dfs = []
for i in range(len(err_images)):
    err_image = (err_images[i].squeeze() * 255).astype(np.uint8)
    err_pred = err_preds[i]
    err_truth = err_truths[i]
    df = pl.DataFrame(err_image)
    df = (
        df.with_columns(pl.arange(0, df.height).alias("y"))
        .unpivot(index="y")
        .rename({"variable": "x", "value": "value"})
    )
    df = df.with_columns(
        pl.col("x").str.replace(r"column_", "").cast(pl.Int64),
        pred=pl.lit(f"pred={err_pred}"),
        truth=pl.lit(f"truth={err_truth}"),
        idx=pl.lit(f"image={i}"),
    )
    df = df.with_columns(
        pl.concat_str(
            [pl.col("idx"), pl.col("pred"), pl.col("truth")], separator="\t"
        ).alias("title"),
    )
    err_dfs.append(df)
assert len(err_dfs) == len(err_images)
print(len(err_dfs))

# %%
selected_idxs = np.random.choice(
    range(len(err_images)), size=16, replace=False
)
selected_images = [err_images[i] for i in selected_idxs]
err_plot_dfs = [err_dfs[i] for i in selected_idxs]
err_plot_df = pl.concat(err_plot_dfs)
print(err_plot_df.shape)
print(err_plot_df["title"].unique())


# %%
plots = []
for err_image in selected_images:
    p = figure(width=144, height=144)
    p.image(
        image=[np.flipud(err_image.squeeze() * 255).astype(np.uint8)],
        x=0,
        y=0,
        dw=28,
        dh=28,
        palette="Greys256",
    )
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.outline_line_color = None
    plots.append(p)
plots = list(np.array(plots).reshape(4, 4))

# %%
gp = gridplot(plots)
show(gp)

# %%
alt.data_transformers.disable_max_rows()
heatmap = (
    alt.Chart(err_plot_df)
    .mark_rect()
    .encode(
        x="x:O",
        y="y:O",
        color=alt.Color(
            "value:Q",
            scale=alt.Scale(domain=[0, 255], range=["black", "white"]),
        ),
    )
    .properties(width=144, height=144)
    .facet(facet="title:O", columns=4)
    .configure_axis(disable=True)
    .configure_view(stroke=None)
    .configure_legend(disable=True)
)
heatmap
