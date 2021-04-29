from collections import OrderedDict

LABELS = OrderedDict(
    {
        -1: "No prediction",
        0: "Surprise",
        1: "Fear",
        2: "Disgust",
        3: "Happiness",
        4: "Sadness",
        5: "Anger",
        6: "Neutral",
    }
)
PLOT_RC = {
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "font.family": ["serif"],
    "grid.color": "gainsboro",
    "grid.linestyle": "-",
    "patch.edgecolor": "none",
}
