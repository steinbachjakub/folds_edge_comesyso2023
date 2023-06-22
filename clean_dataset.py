"""
Script used to restructure the original folder with dataset. Nothing more...
"""

from pathlib import Path

PATH_DATA = Path("data")
PATH_DATA.joinpath("images").mkdir(exist_ok=True)
PATH_DATA.joinpath("labels").mkdir(exist_ok=True)

all_images = list(PATH_DATA.joinpath("fold0").glob("*/*.jpg"))
all_labels = list(PATH_DATA.joinpath("fold0").glob("*/*.txt"))

for image in all_images:
    image.rename(PATH_DATA.joinpath("images", image.name))

for label in all_labels:
    label.rename(PATH_DATA.joinpath("labels", label.name))

