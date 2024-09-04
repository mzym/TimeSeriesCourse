import numpy as np
import matplotlib.pyplot as plt


# Визуализация сниппетов
def plot_snippets(ts, snippets):
    with plt.rc_context(
        {
            "lines.linewidth": 2,
            "font.size": 24,
        }
    ):
        fig, (ax_main, ax_labels) = plt.subplots(
            2, figsize=(16, 6), gridspec_kw={"height_ratios": [16, 2]}
        )
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        ax_main.plot(ts, color="gray")

        m = len(snippets[0][0])
        print(m)
        snippets_indices = snippets[1]
        for i, snippet_start in enumerate(snippets_indices):
            snippet_end = snippet_start + m
            ax_main.plot(
                np.arange(snippet_start, snippet_end),
                ts[snippet_start:snippet_end],
                c=color_cycle[i],
                label=f"Snippet {i}: {snippets[3][i]:.2f}",
            )

        labels = np.zeros_like(ts)
        snippets_regimes = snippets[5]
        for regime in snippets_regimes:
            labels[regime[1] : regime[2]] = regime[0]

        img = ax_labels.imshow([range(len(color_cycle))], cmap="tab10", aspect="auto")
        img.set_data([labels])

        ax_main.set_xlim(0, len(ts))
        ax_labels.axis("off")
        ax_main.legend(prop={"size": 16}, loc="upper right")
        plt.tight_layout()
        plt.show()
    return ax_main
