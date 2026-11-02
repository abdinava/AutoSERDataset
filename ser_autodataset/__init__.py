# __init__ is a special filename python recognizes.
# it flags the file as a package, so when you import ser_autodataset
# it runs this file first.
#lets you use each function without calling the full path.
#instead, you can call simply the function rather than .file then function

#list important tools from package:

from .config import DatasetConfig
from .pipeline import process_media_file, create_emotion_labeled_dataset
from .batch import process_zip_of_media

#DatasetConfig: settings.
#process_media_file: process one video or audio file into clips.
#create_emotion_labeled_dataset: labels those clips with emotions and zips!
#process_zaip_ofmedia: does everything but for a zip folder.

# list of what this package offers (public stuff list fr fr):

__all__ = [
    "DatasetConfig",
    "process_media_file",
    "create_emotion_labeled_dataset",
    "process_zip_of_media",
]

# Version
__version__ = "5.0.0"
