# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

st.set_page_config(page_title="Binomial Trees", layout="wide")

# CSS for responsive iframe scaling
st.markdown("""
<style>
/* Make the entire app responsive */
.main .block-container {
    max-width: 100% !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Scale content to fit iframe */
.stApp {
    transform-origin: top left;
    width: 100%;
}

/* Ensure plots scale properly */
.js-plotly-plot {
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

plotly_template = pio.templates["simple_white"]
colors = plotly_template.layout.colorway
blue, red, green, purple, orange = colors[:5]
red = colors[1]
green = colors[2]
purple = colors[3]
orange = colors[4]




def stockTree(S, u, n):
    return [[S * u ** (t - 2 * i) for i in range(t + 1)] for t in range(n + 1)]


def europeanTree(S, K, r, u, n, kind):
    def f(S):
        if kind == "call":
            return np.maximum(np.array(S) - K, 0)
        else:
            return np.maximum(K - np.array(S), 0)

    d = 1 / u
    p = (1 + r - d) / (u - d)
    disc = 1 / (1 + r)
    ST = [S * u ** (n - 2 * i) for i in range(n + 1)]
    x = f(ST)
    lst = [x]
    while len(x) > 1:
        x = disc * (p * x[:-1] + (1 - p) * x[1:])
        lst.insert(0, x)
    return [list(x) for x in lst], p


def americanTree(S, K, r, u, n, kind):
    def f(S):
        if kind == "call":
            return np.maximum(np.array(S) - K, 0)
        else:
            return np.maximum(K - np.array(S), 0)

    d = 1 / u
    p = (1 + r - d) / (u - d)
    disc = 1 / (1 + r)
    ST = [S * u ** (n - 2 * i) for i in range(n + 1)]
    x = f(ST)
    lst = [x]
    while len(x) > 1:
        x0 = disc * (p * x[:-1] + (1 - p) * x[1:])
        t = len(x0) - 1
        St = [S * u ** (t - 2 * i) for i in range(t + 1)]
        x = np.maximum(x0, f(St))
        lst.insert(0, x)
    return [list(x) for x in lst], p


def treePlot(tree, kind):
    color = green if kind == "option" else blue
    string = "$%{y:,.2f}<extra></extra>"
    spliced = []
    for a, b in zip(tree[1:], tree[:-1]):
        x = []
        for i in range(len(a)):
            x.append(a[i])
            try:
                x.append(b[i])
            except:
                pass
        spliced.append(x)
    fig = go.Figure()
    for i in range(len(tree) - 1):
        x = [1, 0, 1]
        for j in range(i):
            x.append(0)
            x.append(1)
        x = np.array(x) + i
        y = spliced[i]
        trace = go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            hovertemplate=string,
            marker=dict(size=12, color=color),
            line=dict(color=color),
            showlegend=False
        )
        fig.add_trace(trace)
    fig.update_layout(
        xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        showlegend=False
    )
    fig.update_xaxes(title="Time")
    fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.0f")
    return fig

def figtbl(kind, S, K, r, u, n):
    r /= 100
    u = 1 + u / 100
    tree = stockTree(S, u, n)
    fig_stock = treePlot(tree, kind="stock")
    fig_stock.update_yaxes(title="Underlying Price")
    x = kind.split(" ")
    Tree = europeanTree if x[0] == "European" else americanTree
    tree, prob = Tree(S, K, r, u, n, x[1])
    value = tree[0][0]
    fig_option = treePlot(tree, kind="option")
    if x[1] == "put":
        fig_option.update_yaxes(autorange="reversed")
    fig_option.update_yaxes(title=kind.title() + " Value")
   
    return fig_stock, fig_option



# Top control area
with st.container():
    col0, col1, col2, col3 = st.columns(4)
    
    with col0:
        option_type = st.radio("Option Type", ["Call", "Put"])
    with col1:
        exercise_type = st.radio("Exercise Style", ["European", "American"])
        
    with col2:
        S = st.number_input("Initial Stock Price ($)", min_value=1.0, value=100.0, step=1.0)
        K = st.number_input("Strike Price ($)", min_value=1.0, value=100.0, step=1.0)
    
    with col3:
        r = st.number_input("Interest Rate (%)", min_value=0.0, value=1.0, step=0.1)
        u = st.number_input("Up Step (%)", min_value=0.1, value=5.0, step=0.1)
    
    col1, col2, col3 = st.columns(3)
    with col2:
        n = st.number_input("Number of Steps", min_value=1, value=3, step=1)

kind = f"{exercise_type} {option_type.lower()}"
tree = stockTree(S, 1 + u/100, n)
fig_stock = treePlot(tree, kind="stock")
fig_stock.update_yaxes(title="Underlying Price")

Tree = europeanTree if exercise_type == "European" else americanTree
tree, prob = Tree(S, K, r/100, 1 + u/100, n, option_type.lower())
value = tree[0][0]
fig_option = treePlot(tree, kind="option")
if option_type.lower() == "put":
    fig_option.update_yaxes(autorange="reversed")
fig_option.update_yaxes(title=kind.title() + " Value")

col1, col2 = st.columns(2)

with col1:
   st.plotly_chart(fig_stock, use_container_width=True)

with col2:
    st.plotly_chart(fig_option, use_container_width=True)

st.write(f"{kind} value at date 0: ${value:.2f}")
st.write(f"Risk-neutral probability: {prob:.1%}")


