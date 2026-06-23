from matplotlib import font_manager
import matplotlib
import matplotlib.pyplot as plt

fname = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"

font_manager.fontManager.addfont(fname)

prop = font_manager.FontProperties(fname=fname)
family_name = prop.get_name()

plt.rcParams["font.family"] = family_name
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = family_name

matplotlib.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})