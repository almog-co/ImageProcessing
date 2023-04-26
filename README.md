# ImageProcessing
Repo for ImageProcessing Barcode Project

## Installation
1. Clone the repo `https://github.com/almog-co/ImageProcessing.git` into a directory on your computer and navigate to it.
2. Setup the `venv` for the dependencies by running `python3 -m venv env`
3. Actiavte the `venv` <br/>
   **For Mac/Linux:** <br/>
   run `source env/bin/activate` <br/> <br/>
   **For Windows:** <br/>
   run `env\Scripts\activate.bat` <br/>
 4. Download all dependencies by running `pip3 install -r requirements.txt`
 5. Now run the code using `python3 decode.py`
   
## Usage
For demo purposes there are two files I recommend using to directly interact with the color codes. `decoding.py` and `encoding.py` but feel free to look around.

**I include a built-in sample image called `colorsample.png`**. Feel free to use this one for a quick demo or upload your own!

### Decoding
For decoding color codes, run 
```
python3 decode.py image.png
```
You can also add `-v` or `--verbose` for verbose mode to see the intermediate results.
```
python3 decode.py image.png -v
```

### Encoding
For encoding color codes, run
```
python3 encode.py "Hello World Color Version!!!"  
```
Like before, you can add `-v` or `--verbose` to see the intermediate results. You can also add `-o` or `--output` to specify the output path. Otherwise, it saves the output file to `code.png` in the same directory. 
```
python3 encode.py "Hello World Color Version!!!" -v -o helloworld.png 
```
