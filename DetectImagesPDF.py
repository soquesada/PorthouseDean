# STEP 1
# import libraries
import fitz
import io
from PIL import Image
import os
from tqdm import tqdm
  
# STEP 2
# file path you want to extract images from
file = "C:\\Users\sophi\OneDrive\Desktop\PorthouseDean\\assets\MERGED 6.pdf"

# workdir = "your_folder"

doc = fitz.Document(file)

for i in tqdm(range(len(doc)), desc="pages"):
    for img in tqdm(doc.get_page_images(i), desc="page_images"):
        xref = img[0]
        image = doc.extract_image(xref)
        pix = fitz.Pixmap(doc, xref)
        pix.save("%s_p%s-%s.png" % (file[:-4], i, xref))

# for each_path in os.listdir(workdir):
#     if ".pdf" in each_path:
#         doc = fitz.Document((os.path.join(workdir, each_path)))

#         for i in tqdm(range(len(doc)), desc="pages"):
#             for img in tqdm(doc.get_page_images(i), desc="page_images"):
#                 xref = img[0]
#                 image = doc.extract_image(xref)
#                 pix = fitz.Pixmap(doc, xref)
#                 pix.save(os.path.join(workdir, "%s_p%s-%s.png" % (each_path[:-4], i, xref)))
                
print("Done!")


# open the file
# pdf_file = fitz.Document(file)
  
# # STEP 3
# # iterate over PDF pages
# for page_index in range(len(pdf_file)):
  
#     # get the page itself
#     page = pdf_file[page_index]
#     image_list = page.getImageList()
  
#     # printing number of images found in this page
#     if image_list:
#         print(
#             f"[+] Found a total of {len(image_list)} images in page {page_index}")
#     else:
#         print("[!] No images found on page", page_index)
#     for image_index, img in enumerate(page.getImageList(), start=1):
  
#         # get the XREF of the image
#         xref = img[0]
  
#         # extract the image bytes
#         base_image = pdf_file.extractImage(xref)
#         image_bytes = base_image["image"]
  
#         # get the image extension
#         image_ext = base_image["ext"]