# Waggle dance annotator tool

This tool provides a GUI interface to annotate videos with the location, direction, and timing of honey bee waggle dances in an observation hive.

![](https://github.com/BioroboticsLab/bb_main/blob/master/images/waggle_annotator_tool.png?raw=True)

## Installation

### Conda + pip

```bash
conda create -n waggle-annotator python=3.12 pip -y
conda activate waggle-annotator
python -m pip install git+https://github.com/BioroboticsLab/bb_waggledance_annotator.git
```

note:  Python 3.11 is also fine.

## Note
The version 'dance_analysis' uses OpenCV for video annotation, and has a memory problem on Mac.  The updated version 'dance_analysis_2' fixes this problem and adds additional features, using a Tkinter interface for annotation

## Usage

After installation, start a new terminal in order to reload settings.  Then you can start the program from a terminal by running:
```bash
dance_analysis_2
```

By default the tool opens the current directory to select videos for annotation.  Optionally the '-p' parameter can be used to specify a directory to open, for example
```bash
dance_analysis_2 -p /path/to/directory
```

See the key mappings and instructions shown in the GUI for how to navigate the videos and mark annotations.
