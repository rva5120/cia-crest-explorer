# Run get_text.py script for every file in the JPRS folder
for file in JPRS/*
do
	python get_text.py "$file"
done
