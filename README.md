<div>
    <h1 align="center">Tube Color Detector<p align="center" dir="auto"></p></h1>
<div>

A Python script that takes the image sample(shown below) as input and gives the colors in the tubes in order in image_name_analysis.json

### Sample Input Images

<div allign = "center">
    <img align="center" src ="assets/bottle0.jpeg" width ="200">
    <img align="center" src ="assets/bottle1.jpeg" width ="200">
    <img align="center" src ="assets/bottle2.jpeg" width ="200">
</div>

### Sample Output

For the image `assets/bottle0.jpeg` the `bottle0_analysis.json` would be as follows:

```json
{
  "tube0": [
    "red",
    "tan",
    "rose",
    "purple"
  ],
  "tube1": [
    "blue",
    "white",
    "rose",
    "orange"
  ],
  "tube2": [
    "blue",
    "red",
    "cyan",
    "lime"
  ],
.....and so on.....
```

### ⚠️ Prerequisites

1. `git` installed and properly setup.

2. The latest `python` or `anaconda` installed (preferrably python).

3. `pip` should be working.

4. You should either be on Linux or Windows.

5. You can you VScode terminal or just rawdawg it via terminal in linux.

### 📁 File Structure

- `main.py` - The main program.

- `modules/` - Contains modules for the `main.py` script.

- `requirements.txt` - The dependencies of the program.

- `assets/` - Contains the sample input for the program.

- `README.md` - The file that you're currently reading 😏.

## How to use the program:

Firstly clone the repo via by pasting the commands in the terminal:

```git
git clone https://github.com/jack-thesparrow/tube-color-detect.git
```

or if you have git SSH setup the you can do:

```shell
git clone git@github.com:jack-thesparrow/tube-color-detect.git
```

Now go inside the directory via:
=======

Now go inside the directory via:

```shell
cd tube-color-detect
```

All the dependencies are listed in the requirements.txt. Install them by running the following command in terminal.

```shell
pip install -r requirements.txt
```

#### Now you can run the script using:

```shell
python main.py <image_name.png>
```

<I>The saved output will be stored in <B>image_name_analysis.json</B></I>

### 🌚 Our Team (Core Contributors)

- [Rahul Tudu](https://github.com/jack-thesparrow) 

- [Prince Patel](https://github.com/princepatel1526)

- [Dev Varshini](https://github.com/varshi06-maker)

- [Satya Keertika](https://github.com/Satyakeerthika07)
