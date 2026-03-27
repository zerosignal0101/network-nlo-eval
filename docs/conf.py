"""Sphinx configuration."""

project = "Network NLO Eval"
author = "Xiangrong Li"
copyright = "2026, Xiangrong Li"
extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
    "sphinx_click",
    "myst_parser",
]
# Sphinx-Gallery 配置
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # 你的 Python 示例脚本存放目录
    "gallery_dirs": "_examples",  # 生成的 HTML 和 Notebook 存放目录
    "filename_pattern": r"plot_",  # 只有以 plot_ 开头的脚本才会被执行并抓取图表
}
autodoc_typehints = "description"
html_theme = "furo"
