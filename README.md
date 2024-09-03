# Waggle dance annotator tool

This tool provides a GUI interface to annotate videos with the location, direction, and timing of honey bee waggle dances in an observation hive.

![](https://github.com/BioroboticsLab/bb_main/blob/master/images/waggle_annotator_tool.png?raw=True)

## Installation

```bash
pip install git+https://github.com/BioroboticsLab/dance_analysis.git
```

## Usage

After installation, start a new terminal in order to reload settings.  Then you can start the program from a terminal by running:
```bash
dance_analysis
```

By default the tool opens the current directory to select videos for annotation.  Optionally the '-p' parameter can be used to specify a directory to open, for example
```bash
dance_analysis -p /path/to/directory
```

See the key mappings and instructions shown in the GUI for how to navigate the videos and mark annotations.

