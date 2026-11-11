"""BSD3513 Lab 5 - App 2: Sigmoid activation function visualiser (Streamlit)."""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Sigmoid Visualiser", page_icon="📈", layout="wide")
st.title("📈 Sigmoid")
st.latex(r"\sigma(x)=\frac{1}{1+e^{-x}}\qquad \sigma'(x)=\sigma(x)\,(1-\sigma(x))")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)


# ---- Sidebar controls ----
st.sidebar.header("Controls")
x_min, x_max = st.sidebar.slider("x-axis range", -10.0, 10.0, (-8.0, 8.0), 0.5)
x_in = st.sidebar.slider("Input value x", float(x_min), float(x_max), 1.0, 0.1)
show_grad = st.sidebar.checkbox("Show derivative", True)

x = np.linspace(x_min, x_max, 400)
col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, sigmoid(x), label="Sigmoid(x)", color="tab:green", lw=2)
    if show_grad:
        ax.plot(x, sigmoid_grad(x), "--", label="Derivative", color="tab:orange")
    ax.scatter([x_in], [sigmoid(x_in)], color="red", zorder=5, label="Your input")
    ax.axhline(0.5, color="grey", lw=0.5, ls=":")
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("x (weighted sum + bias)")
    ax.set_ylabel("Output")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

with col2:
    st.metric("Sigmoid(x)", f"{sigmoid(x_in):.4f}")
    st.metric("Derivative", f"{sigmoid_grad(x_in):.4f}")
    st.markdown(
        "**Key points**\n"
        "- Output range: (0, 1), read as a probability\n"
        "- Used for binary classification output layers\n"
        "- Max gradient is 0.25, so deep stacks suffer vanishing gradients\n"
        "- Not zero-centred"
    )
