from matplotlib import font_manager
import matplotlib.pyplot as plt

fname = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"

font_manager.fontManager.addfont(fname)

prop = font_manager.FontProperties(fname=fname)
family_name = prop.get_name()

plt.rcParams["font.family"] = family_name
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = family_name