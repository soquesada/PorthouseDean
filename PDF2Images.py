import fitz  # PyMuPDF
import os
import glob
from pathlib import Path

# file_path = "my_file.pdf"
workdir = "assets\\"
home = Path(__file__).parent.absolute()
savedir = Path(home, "assets", "unlabelled")

# doc = fitz.Document(file_path)  # Open document
pdf_files = glob.glob(os.path.join(workdir, "*.pdf"))

# for each_path in os.listdir(os.path.join(workdir, "*.pdf")):
for each_path in pdf_files:
    doc = fitz.Document(each_path)
    file_name = os.path.splitext(os.path.basename(each_path))[0]
    print("opened")
    i = 1
    
    for page in doc:
        print(i)
        pix = page.get_pixmap()  # render page to an image
        
        pix.save(savedir.joinpath(f"{file_name}_page_{i}.png"))
        i += 1
