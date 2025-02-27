import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import numpy as np


def process_and_plot_stacked_bar(dfs):
    result_dicts = []
    required_keys = ['A', 'B', 'M', 'X', 'Z']

    for real_df in dfs:
        percentages = real_df['category'].value_counts(normalize=True) * 100
        percentages_dict = {key: percentages.get(key, 0.0) for key in required_keys}
        result_dicts.append(percentages_dict)

    categories = ['A', 'B', 'M', 'X', 'Z']
    colors = ['#A3D8A0', '#4C8C99', '#8BBEE8', '#F8D76E', '#E55353']
    categories_legend = [
        'Single label image, original label is correct',
        'Single label image, original label is incorrect, full agreement on correction',
        'Multilabel images',
        'Single label image, inconsistent label corrections',
        'Ambiguous, no agreement on the label'
    ]

    num_columns = len(result_dicts)
    category_values = {cat: [rd.get(cat, 0) for rd in result_dicts] for cat in categories}
    stacked_data = [category_values[cat] for cat in categories]

    fig, ax = plt.subplots(figsize=(15, 5), dpi=300)
    bottom_values = np.zeros(num_columns)
    total_values = np.sum(stacked_data, axis=0)
    max_value = np.max(total_values)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.7, alpha=0.7, color='gray')

    for idx, (cat, color) in enumerate(zip(categories, colors)):
        ax.bar(range(num_columns), stacked_data[idx], bottom=bottom_values, label=categories_legend[idx], color=color)
        bottom_values += np.array(stacked_data[idx])

    ax.set_ylim(0, max_value)
    ax.set_yticks(np.linspace(0, max_value, 6))
    ax.set_yticklabels([f'{int((y / max_value) * 100)}%' for y in np.linspace(0, max_value, 6)], fontsize=18)

    ax.set_xticks([])
    ax.set_xticklabels([])

    names = ['ImageNet Multilabel', 'Contextualizing Progress', 'ImageNet Real', 'Label Errors']
    for i in range(num_columns):
        ax.text(i, -max_value * 0.05, names[i], ha='center', va='center', fontsize=18, color='black')

    plt.tight_layout()
    plt.savefig('stacked_bar_chart.svg', format='svg', transparent=True)
    plt.show()


def plot_venn(labels, set_1, set_2, set_3=None, title=None, font=7):
    plt.figure(dpi=300, figsize=(2, 2))
    if title:
        plt.title(title, fontsize=font)

    venn_func = venn3 if set_3 else venn2
    sets = [set_1, set_2] if not set_3 else [set_1, set_2, set_3]
    colors = ['#F8D76E', '#4C8C99', '#A3D8A0'][:len(sets)]

    venn = venn_func(sets, set_labels=labels, set_colors=colors)

    for label in venn.set_labels:
        label.set_fontsize(font)

    for subset in venn.subset_labels:
        if subset and subset.get_text() == '0':
            subset.set_text('')
        elif subset:
            subset.set_fontsize(font)

    for patch in venn.patches:
        if patch:
            patch.set_alpha(0.8)

    plt.show()
