import doc2text
import sys
import os

input_file = sys.argv[1]

doc = doc2text.Document(lang="eng")
doc.read(input_file)
doc.process()
doc.extract_text()
text = doc.get_text()

filename, filename_ext = os.path.splitext(input_file)
output_file = filename+".txt"
out = open(output_file, "w")
out.write(text)
